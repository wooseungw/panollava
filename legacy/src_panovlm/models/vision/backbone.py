"""Vision backbone wrappers for Panorama VLM."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import math

import torch
import torch.nn as nn
from transformers import AutoModel


class VisionBackbone(nn.Module):
    def __init__(
        self,
        vision_name: str,
        use_vicreg_norm: bool = False,
        backbone_type: str = "hf",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        input_key: str = "pixel_values",
        output_key: str = "last_hidden_state",
        forward_method: Optional[str] = None,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.input_key = input_key
        self.output_key = output_key
        self.forward_method = forward_method

        if backbone_type == "hf":
            encoder = AutoModel.from_pretrained(vision_name, trust_remote_code=True, **backbone_kwargs)
            if hasattr(encoder, "vision_model"):
                encoder = encoder.vision_model
        elif backbone_type == "timm":
            import timm
            encoder = timm.create_model(vision_name, pretrained=True, **backbone_kwargs)
        elif backbone_type == "torchhub":
            repo, model_name = vision_name.split(":", 1)
            encoder = torch.hub.load(repo, model_name, **backbone_kwargs)
        elif backbone_type == "module":
            import importlib
            module_path, attr_name = vision_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            encoder_cls = getattr(module, attr_name)
            encoder = encoder_cls(**backbone_kwargs)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.encoder = encoder
        self.hidden_size = int(hidden_size) if hidden_size is not None else self._infer_hidden_size(self.encoder)
        self.use_vicreg_norm = use_vicreg_norm
        self.norm = nn.LayerNorm(self.hidden_size) if use_vicreg_norm else nn.Identity()

    @staticmethod
    def _infer_hidden_size(vision_model: nn.Module) -> int:
        config = getattr(vision_model, "config", None)
        for key in ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]:
            if config is not None and hasattr(config, key):
                return getattr(config, key)
            if hasattr(vision_model, key):
                return getattr(vision_model, key)
        for key in ["num_features", "feature_dim"]:
            if hasattr(vision_model, key):
                return getattr(vision_model, key)
        raise AttributeError("비전 모델의 은닉 차원 크기를 찾을 수 없습니다")

    def normalize_inputs(self, pixel_values: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> Tuple[int, int, torch.Tensor]:
        if pixel_values.ndim == 5:
            batch_size, num_views, _, _, _ = pixel_values.shape
        elif pixel_values.ndim == 4:
            batch_size, _, _, _ = pixel_values.shape
            num_views = 1
            pixel_values = pixel_values.unsqueeze(1)
        else:
            raise ValueError(f"pixel_values shape invalid: {pixel_values.shape}")
        if target_dtype is not None and pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(target_dtype)
        return batch_size, num_views, pixel_values

    def extract_features(self, normalized_pixels: torch.Tensor) -> torch.Tensor:
        flattened_pixel_values = normalized_pixels.view(normalized_pixels.size(0) * normalized_pixels.size(1), *normalized_pixels.shape[2:])
        if self.forward_method == "forward_features" and hasattr(self.encoder, "forward_features"):
            vision_output = self.encoder.forward_features(flattened_pixel_values)
        else:
            call_kwargs = {self.input_key: flattened_pixel_values}

            # SigLIP expects interpolate_pos_encoding=True when input resolution differs.
            try:
                model_type = getattr(self.encoder, "config", None)
                model_type = getattr(model_type, "model_type", "")
                if model_type and "siglip" in model_type:
                    call_kwargs["interpolate_pos_encoding"] = True
            except Exception:
                pass

            try:
                vision_output = self.encoder(**call_kwargs, return_dict=True)
            except TypeError:
                vision_output = self.encoder(**call_kwargs)

        if isinstance(vision_output, torch.Tensor):
            vision_hidden_states = vision_output
        elif isinstance(vision_output, (tuple, list)) and vision_output:
            vision_hidden_states = vision_output[0]
        elif hasattr(vision_output, self.output_key):
            vision_hidden_states = getattr(vision_output, self.output_key)
        elif isinstance(vision_output, dict) and self.output_key in vision_output:
            vision_hidden_states = vision_output[self.output_key]
        else:
            raise ValueError(f"Unsupported vision output type for key '{self.output_key}'")

        if vision_hidden_states.dim() == 4:
            if vision_hidden_states.shape[1] == self.hidden_size:
                vision_hidden_states = vision_hidden_states.flatten(2).transpose(1, 2)
            elif vision_hidden_states.shape[-1] == self.hidden_size:
                vision_hidden_states = vision_hidden_states.view(vision_hidden_states.size(0), -1, self.hidden_size)
            else:
                vision_hidden_states = vision_hidden_states.flatten(2).transpose(1, 2)

        return self.norm(vision_hidden_states)

    def forward(self, pixel_values: torch.Tensor, target_dtype: Optional[torch.dtype] = None):
        batch_size, num_views, normalized_pixels = self.normalize_inputs(pixel_values, target_dtype)
        vision_hidden_states = self.extract_features(normalized_pixels)
        return {
            "vision_features": vision_hidden_states,
            "batch_size": batch_size,
            "num_views": num_views,
            "device": normalized_pixels.device,
        }

    def has_cls_token(self, vision_output: torch.Tensor) -> bool:
        for attr in ("cls_token", "class_token", "class_embedding"):
            if hasattr(self.encoder, attr):
                return True
        seq_len = vision_output.size(1)
        if seq_len > 1:
            if math.isqrt(seq_len) ** 2 == seq_len:
                return False
            if math.isqrt(seq_len - 1) ** 2 == (seq_len - 1):
                return True
        return False
