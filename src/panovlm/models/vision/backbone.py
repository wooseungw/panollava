"""Vision backbone wrappers for Panorama VLM."""

from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
from transformers import AutoModel


class VisionBackbone(nn.Module):
    def __init__(self, vision_name: str, use_vicreg_norm: bool = False):
        super().__init__()
        encoder = AutoModel.from_pretrained(vision_name, trust_remote_code=True)
        if hasattr(encoder, "vision_model"):
            encoder = encoder.vision_model
        self.encoder = encoder
        self.hidden_size = self._infer_hidden_size(self.encoder)
        self.use_vicreg_norm = use_vicreg_norm
        self.norm = nn.LayerNorm(self.hidden_size) if use_vicreg_norm else nn.Identity()

    @staticmethod
    def _infer_hidden_size(vision_model: nn.Module) -> int:
        for key in ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]:
            if hasattr(vision_model.config, key):
                return getattr(vision_model.config, key)
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
        vision_output = self.encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state
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
