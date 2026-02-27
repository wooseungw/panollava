"""Baseline LoRA finetuning trainer for commercial VLMs.

Uses HuggingFace Trainer + PEFT LoRA on models loaded via
:class:`BaselineModelRegistry`. Ported from the
``LoRAAblationRunner`` in legacy/root_scripts/vlm_finetune_and_eval.py.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.utils.data
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from .config import BaselineConfig, PanoAdaptConfig, PanoViewConfig
from .models import BaselineModelRegistry

# Allow large panorama images without PIL decompression bomb warnings.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto batch-size (YOLO-style GPU memory profiling)
# ---------------------------------------------------------------------------

def autobatch(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn,
    fraction: float = 0.60,
    default_batch_size: int = 1,
) -> int:
    if not torch.cuda.is_available():
        logger.warning("AutoBatch: CUDA not available, using batch_size=%d", default_batch_size)
        return default_batch_size

    import numpy as np

    device = torch.device("cuda")
    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / gb

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.to(device).train()

    allocated_after_model = torch.cuda.memory_allocated(device) / gb
    free = total - allocated_after_model

    logger.info(
        "AutoBatch: %s %.1fG total, %.1fG model, %.1fG free (%.0f%% target)",
        props.name, total, allocated_after_model, free, fraction * 100,
    )

    batch_sizes = [1, 2, 4, 8, 16]
    mem_usage = []

    for bs in batch_sizes:
        if bs > len(dataset):
            break
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            batch = collate_fn([dataset[i % len(dataset)] for i in range(bs)])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()
            peak = torch.cuda.max_memory_allocated(device) / gb
            mem_usage.append((bs, peak))
            model.zero_grad(set_to_none=True)
            del batch, outputs, loss
            torch.cuda.empty_cache()
            logger.info("  batch %2d -> %.2fG peak", bs, peak)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                logger.info("  batch %2d -> OOM", bs)
                break
            raise

    model.zero_grad(set_to_none=True)
    model.cpu()
    torch.cuda.empty_cache()

    if len(mem_usage) < 2:
        logger.warning("AutoBatch: insufficient data, using batch_size=%d", default_batch_size)
        return default_batch_size

    xs, ys = zip(*mem_usage)
    p = np.polyfit(xs, ys, deg=1)
    optimal = int((free * fraction - p[1]) / p[0])

    oom_limit = batch_sizes[len(mem_usage)] if len(mem_usage) < len(batch_sizes) else batch_sizes[-1] * 2
    optimal = min(optimal, oom_limit - 1)
    optimal = max(optimal, 1)

    logger.info("AutoBatch: optimal batch_size = %d (%.1fG / %.1fG, %.0f%%)", optimal, np.polyval(p, optimal), total, fraction * 100)
    return optimal


def autobatch_generate(
    model: torch.nn.Module,
    sample_inputs_fn,
    max_new_tokens: int = 128,
    fraction: float = 0.85,
    default_batch_size: int = 1,
) -> int:
    if not torch.cuda.is_available():
        return default_batch_size

    import numpy as np

    device = next(model.parameters()).device
    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / gb

    torch.cuda.empty_cache()
    allocated_model = torch.cuda.memory_allocated(device) / gb
    free = total - allocated_model

    logger.info(
        "AutoBatch (eval): %s %.1fG total, %.1fG free (%.0f%% target)",
        props.name, total, free, fraction * 100,
    )

    batch_sizes = [1, 2, 4, 8, 16]
    mem_usage = []

    for bs in batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            inputs = sample_inputs_fn(bs)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=min(max_new_tokens, 16), do_sample=False)
            peak = torch.cuda.max_memory_allocated(device) / gb
            mem_usage.append((bs, peak))
            del inputs
            torch.cuda.empty_cache()
            logger.info("  batch %2d -> %.2fG peak", bs, peak)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.info("  batch %2d -> OOM", bs)
                break
            raise

    if len(mem_usage) < 2:
        return default_batch_size

    xs, ys = zip(*mem_usage)
    p = np.polyfit(xs, ys, deg=1)
    optimal = int((free * fraction - p[1]) / p[0])

    oom_limit = batch_sizes[len(mem_usage)] if len(mem_usage) < len(batch_sizes) else batch_sizes[-1] * 2
    optimal = min(optimal, oom_limit - 1)
    optimal = max(optimal, 1)

    logger.info("AutoBatch (eval): optimal batch_size = %d", optimal)
    return optimal


# ---------------------------------------------------------------------------
# bf16 availability check (transformers compat shim)
# ---------------------------------------------------------------------------

try:
    from transformers.utils import is_torch_bf16_gpu_available as _is_bf16_supported
except ImportError:
    try:
        from transformers.utils import is_bfloat16_supported as _is_bf16_supported  # type: ignore[assignment]
    except ImportError:

        def _is_bf16_supported() -> bool:  # type: ignore[misc]
            if not torch.cuda.is_available():
                return False
            return getattr(torch.cuda, "is_bf16_supported", lambda: False)()


# ---------------------------------------------------------------------------
# VLM Dataset
# ---------------------------------------------------------------------------


class VLMDataset(torch.utils.data.Dataset):
    """CSV-backed dataset that yields processor-ready dicts.

    Supports single-image (default) and multi-image (pano_view) modes.
    When ``pano_view_config`` is provided, each sample produces multiple
    perspective tile views from the ERP panorama image based on the
    configured strategy (anyres_e2p, cubemap, or pinhole).
    """

    def __init__(
        self,
        csv_path: str,
        processor: Any,
        tokenizer: Any,
        model_type: str,
        image_column: str = "url",
        instruction_column: str = "instruction",
        response_column: str = "response",
        max_samples: Optional[int] = None,
        anyres_config: Optional["PanoViewConfig"] = None,
        pano_view_config: Optional["PanoViewConfig"] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples > 0:
            self.df = self.df.head(max_samples)

        self.processor = processor
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.image_column = image_column
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.pano_view_config = pano_view_config or anyres_config

    def __len__(self) -> int:
        return len(self.df)

    # ---- helpers ----------------------------------------------------------

    def _load_image(self, row: pd.Series) -> Image.Image:
        path = row.get(self.image_column)
        if path is None:
            raise ValueError(f"CSV missing column '{self.image_column}'")
        img_path = Path(str(path))
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def _load_multi_images(self, row: pd.Series) -> List[Image.Image]:
        """Generate perspective views from ERP panorama based on strategy."""
        from cora.processors.anyres_e2p import build_anyres_from_erp

        erp_img = self._load_image(row)
        cfg = self.pano_view_config
        assert cfg is not None

        strategy = cfg.strategy.lower()

        if strategy == "cubemap":
            hfov, overlap = 90.0, 0.0
        else:
            hfov, overlap = cfg.hfov_deg, cfg.overlap

        pack = build_anyres_from_erp(
            erp_img=erp_img,
            base_size=cfg.base_size,
            tile_render_size=cfg.tile_render_size,
            vit_size=cfg.vit_size,
            hfov_deg=hfov,
            overlap=overlap,
            closed_loop_yaw=cfg.closed_loop_yaw,
            pitch_min=cfg.pitch_min,
            pitch_max=cfg.pitch_max,
        )

        import numpy as np

        def _t2pil(t: torch.Tensor) -> Image.Image:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        pil_views: List[Image.Image] = []
        if cfg.include_global:
            pil_views.append(_t2pil(pack.global_image))
        for i in range(pack.tiles.size(0)):
            pil_views.append(_t2pil(pack.tiles[i]))

        return pil_views

    def _get_text(self, row: pd.Series, column: str, default: str = "") -> str:
        val = str(row.get(column, default))
        if not val or val.lower() == "nan":
            return default
        return val

    @property
    def is_multi_image(self) -> bool:
        return self.pano_view_config is not None

    # ---- per-sample processing --------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt_idx in range(len(self.df)):
            try:
                real_idx = (idx + attempt_idx) % len(self.df)
                row = self.df.iloc[real_idx]
                prompt = self._get_text(row, self.instruction_column, "Describe the image.")
                response = self._get_text(row, self.response_column, "")

                if self.model_type in {"blip2", "blip-2"}:
                    image = self._load_image(row)
                    return self._prepare_seq2seq(image, prompt, response)

                if self.is_multi_image:
                    images = self._load_multi_images(row)
                    return self._prepare_causal_multi(images, prompt, response)

                image = self._load_image(row)
                return self._prepare_causal(image, prompt, response)
            except Exception as e:
                if attempt_idx == 0:
                    logger.warning(f"Skipping idx {idx}: {e}")
                continue
        raise RuntimeError(f"No valid sample found after {len(self.df)} attempts")

    def _prepare_causal(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """Prepare a single-image causal-LM sample."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )

        return {
            "full_text": full_text,
            "prompt_text": prompt_text,
            "image": image,
            "reference_text": response,
        }

    def _prepare_causal_multi(
        self,
        images: List[Image.Image],
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """Prepare a multi-image causal-LM sample (anyres-e2p tiles)."""
        image_entries: List[Dict[str, str]] = [{"type": "image"} for _ in images]
        messages = [
            {
                "role": "user",
                "content": image_entries + [{"type": "text", "text": prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )

        return {
            "full_text": full_text,
            "prompt_text": prompt_text,
            "images": images,
            "reference_text": response,
        }

    def _prepare_seq2seq(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """Prepare an encoder-decoder sample (BLIP-2)."""
        enc = self.processor(images=image, text=response, return_tensors="pt")
        labels = enc["input_ids"].squeeze(0).clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        result: Dict[str, Any] = {"labels": labels}
        for key, value in enc.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)
            elif value is not None:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def _collate_causal(processor: Any, tokenizer: Any, max_length: Optional[int] = None) -> Any:
    """Return a collate function for causal VLMs.

    Handles both single-image (``feature["image"]``) and multi-image
    (``feature["images"]``) samples transparently.
    """

    # NOTE: Do NOT pass max_length/truncation to the processor — it breaks
    # multimodal models (e.g. InternVL) whose processors validate image
    # token counts before truncation.  We truncate manually after processing.
    proc_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}

    def _get_images(f: Dict[str, Any]) -> List[Image.Image]:
        """Extract image(s) from a single feature dict."""
        if "images" in f:
            return f["images"]
        return [f["image"]]

    def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]
        per_sample_images = [_get_images(f) for f in features]

        # Qwen2.5-VL processor expects a flat image list; it matches images
        # to text via the number of <|image_pad|> token blocks in each text.
        flat_images: List[Image.Image] = []
        for imgs in per_sample_images:
            flat_images.extend(imgs)

        full_inputs = processor(
            text=full_texts, images=flat_images, **proc_kwargs,
        )

        # --- Post-process truncation (BEFORE prompt masking) ---
        if max_length is not None:
            full_inputs["input_ids"] = full_inputs["input_ids"][:, :max_length]
            full_inputs["attention_mask"] = full_inputs["attention_mask"][:, :max_length]

        labels = full_inputs["input_ids"].clone()

        for i in range(len(features)):
            single = processor(
                text=[prompt_texts[i]], images=per_sample_images[i],
                return_tensors="pt", padding=False,
            )
            prompt_len = min(single["input_ids"].shape[1], labels.shape[1])
            if prompt_len >= labels.shape[1]:
                logger.warning(
                    "Sample %d: prompt_len (%d) >= max_length (%d). "
                    "No training signal for this sample.",
                    i, single["input_ids"].shape[1], labels.shape[1],
                )
            labels[i, :prompt_len] = -100

        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        result: Dict[str, Any] = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }
        for key in full_inputs:
            if key not in result:
                result[key] = full_inputs[key]
        return result

    return collate_fn


def _collate_seq2seq(tokenizer: Any) -> Any:
    """Return a collate function for encoder-decoder VLMs (BLIP-2)."""

    def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        input_ids_list = [f["input_ids"] for f in features]
        attn_list = [
            f["attention_mask"] for f in features
            if f.get("attention_mask") is not None
        ]

        pad_dict: Dict[str, Any] = {"input_ids": input_ids_list}
        if attn_list:
            pad_dict["attention_mask"] = attn_list
        padded = tokenizer.pad(pad_dict, padding=True, return_tensors="pt")

        labels = tokenizer.pad(
            {"input_ids": [f["labels"] for f in features]},
            padding=True, return_tensors="pt",
        )["input_ids"]
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        batch: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": padded["input_ids"],
            "labels": labels,
        }
        if "attention_mask" in padded:
            batch["attention_mask"] = padded["attention_mask"]
        return batch

    return collate_fn


# ---------------------------------------------------------------------------
# Precision helper
# ---------------------------------------------------------------------------


def _resolve_precision(mp: Optional[str]) -> Dict[str, bool]:
    if mp is None:
        return {"fp16": False, "bf16": False}
    key = mp.lower()
    if key == "bfp16":
        logger.warning("Typo 'bfp16' detected in mixed_precision config; auto-correcting to 'bf16'.")
        key = "bf16"
    if key == "bf16":
        if _is_bf16_supported():
            return {"fp16": False, "bf16": True}
        logger.warning("bf16 not supported; falling back to fp16.")
        return {"fp16": True, "bf16": False}
    if key == "fp16":
        return {"fp16": True, "bf16": False}
    logger.warning("Unknown mixed_precision '%s'; disabling.", mp)
    return {"fp16": False, "bf16": False}


# ---------------------------------------------------------------------------
# BaselineTrainer
# ---------------------------------------------------------------------------


class _SafeTrainer(Trainer):
    """Trainer subclass that handles kwargs incompatible with some model forward() signatures.

    Newer versions of HF Trainer pass ``num_items_in_batch`` to
    ``compute_loss`` which eventually reaches ``model.forward()``.
    Models like T5 (used in BLIP-2) do not accept this kwarg.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        except TypeError as exc:
            if "num_items_in_batch" in str(exc):
                kwargs.pop("num_items_in_batch", None)
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            raise


class _PanoAdaptTrainer(_SafeTrainer):
    """Trainer with PanoAdapt spatial PE and optional DenseCL auxiliary loss.

    Spatial PE (Layer 2): Before each forward pass, computes default
    M-RoPE position_ids via ``get_rope_index``, then shifts the width
    axis for panoramic tile views so adjacent views share continuous
    spatial encodings.

    DenseCL (Layer 3): Registers a forward hook on PatchMerger to extract
    vision features, then computes InfoNCE overlap loss as an auxiliary
    training signal added to the main LM loss.
    """

    def __init__(
        self,
        *args: Any,
        panoadapt_config: PanoAdaptConfig,
        pano_view_config: Optional[PanoViewConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pa_cfg = panoadapt_config
        self._include_global = pano_view_config.include_global if pano_view_config else True
        self._hook: Optional[Any] = None
        self._densecl_loss: Optional[torch.nn.Module] = None

        from cora.baseline.panoadapt import create_vlm_adapter

        model_type = getattr(panoadapt_config, "model_type", "qwen_vl")
        self._adapter = create_vlm_adapter(
            model_type=model_type,
            overlap_ratio=panoadapt_config.overlap_ratio,
            include_global=self._include_global,
        )

        if panoadapt_config.overlap_loss:
            from cora.baseline.panoadapt import VisionFeatureHook, create_panoadapt_loss

            self._hook = VisionFeatureHook()
            self._densecl_loss = create_panoadapt_loss(panoadapt_config)
            self._register_hook(self.model)

    # -- model unwrapping ---------------------------------------------------

    @staticmethod
    def _unwrap_to_rope_model(model: torch.nn.Module) -> torch.nn.Module:
        m = model
        while hasattr(m, "base_model"):
            next_m = m.base_model
            if next_m is m:
                break
            m = next_m
        while hasattr(m, "model") and not hasattr(m, "get_rope_index"):
            next_m = m.model
            if next_m is m:
                break
            m = next_m
        return m

    @staticmethod
    def _unwrap_to_cond_gen(model: torch.nn.Module) -> torch.nn.Module:
        m = model
        while hasattr(m, "base_model"):
            next_m = m.base_model
            if next_m is m:
                break
            m = next_m
        if hasattr(m, "model"):
            next_m = m.model
            if next_m is not m:
                m = next_m
        return m

    # -- hook management ----------------------------------------------------

    def _register_hook(self, model: torch.nn.Module) -> None:
        if self._hook is None:
            return
        cg = self._unwrap_to_cond_gen(model)
        hook_target = self._adapter.get_vision_hook_target() if self._adapter is not None else None
        self._hook.register(cg, hook_target_name=hook_target)
        logger.info("PanoAdapt: VisionFeatureHook registered")

    def cleanup(self) -> None:
        if self._hook is not None:
            self._hook.remove()

    # -- spatial PE ---------------------------------------------------------

    def _apply_pano_widths(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        config: Any,
    ) -> torch.Tensor:
        image_token_id = config.image_token_id
        spatial_merge = config.vision_config.spatial_merge_size
        stride = 1.0 - self._pa_cfg.overlap_ratio

        position_ids = position_ids.clone()

        for batch_idx in range(input_ids.shape[0]):
            is_image = input_ids[batch_idx] == image_token_id
            if not is_image.any():
                continue

            image_positions = is_image.nonzero(as_tuple=True)[0]
            pos = 0
            for view_idx in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[view_idx].tolist()
                llm_w = w // spatial_merge
                n_tokens = t * (h // spatial_merge) * llm_w
                if pos + n_tokens > len(image_positions):
                    break

                view_positions = image_positions[pos : pos + n_tokens]

                # Global view (view_idx=0 when include_global) keeps default positions
                if self._include_global and view_idx == 0:
                    pos += n_tokens
                    continue

                tile_idx = view_idx - (1 if self._include_global else 0)
                pano_shift = int(round(tile_idx * stride * llm_w))
                position_ids[2, batch_idx, view_positions] += pano_shift

                pos += n_tokens

        return position_ids

    # -- compute_loss override ----------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self._pa_cfg.spatial_pe and self._adapter is not None:
            inner = self._unwrap_to_rope_model(model)
            rope_inputs = self._adapter.compute_rope_inputs(inner, inputs)
            position_ids = rope_inputs["position_ids"]
            image_grid_info = inputs.get("image_grid_thw")
            position_ids = self._adapter.modify_position_ids(
                position_ids, inputs["input_ids"], image_grid_info, inner,
            )
            inputs = dict(inputs)
            inputs["position_ids"] = position_ids
            if "rope_deltas" in rope_inputs:
                inputs["rope_deltas"] = rope_inputs["rope_deltas"]

        # -- Clear hook buffer --
        if self._hook is not None:
            self._hook.clear()

        # -- Standard forward + LM loss --
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # -- DenseCL auxiliary loss --
        if self._hook is not None and self._hook.has_features and self._densecl_loss is not None:
            loss = result[0] if isinstance(result, tuple) else result
            densecl_loss = self._compute_densecl(inputs, model)
            if densecl_loss is not None:
                total_loss = loss + self._pa_cfg.overlap_loss_weight * densecl_loss
                logger.debug(
                    "PanoAdapt loss: lm=%.4f densecl=%.4f total=%.4f",
                    loss.item(), densecl_loss.item(), total_loss.item(),
                )
                result = (total_loss, result[1]) if isinstance(result, tuple) else total_loss

        return result

    def _compute_densecl(self, inputs: Dict[str, Any], model: torch.nn.Module) -> Optional[torch.Tensor]:
        assert self._hook is not None and self._densecl_loss is not None
        features = self._hook.get_features()
        if features is None:
            return None

        inner = self._unwrap_to_rope_model(model)
        image_grid_thw = inputs.get("image_grid_thw")

        tile_feats: List[torch.Tensor] = []
        grid_h_val: Optional[int] = None
        grid_w_val: Optional[int] = None

        if image_grid_thw is not None:
            spatial_merge = self._adapter.get_spatial_merge_size(inner)
            num_images = image_grid_thw.shape[0]
            start_view = 1 if (self._include_global and num_images > 1) else 0
            offset = 0
            for vi in range(start_view):
                t, h, w = image_grid_thw[vi].tolist()
                offset += int(t * (h // spatial_merge) * (w // spatial_merge))

            for vi in range(start_view, num_images):
                t, h, w = image_grid_thw[vi].tolist()
                lh, lw = h // spatial_merge, w // spatial_merge
                nt = int(t * lh * lw)
                if offset + nt > features.shape[0]:
                    break
                tile_feats.append(features[offset: offset + nt])
                if grid_h_val is None:
                    grid_h_val, grid_w_val = lh, lw
                offset += nt
        else:
            from cora.baseline.panoadapt import _split_consecutive_groups

            image_token_id = self._adapter.get_image_token_id(inner)
            input_ids = inputs["input_ids"]
            is_image = input_ids[0] == image_token_id
            if not is_image.any():
                return None

            image_positions = is_image.nonzero(as_tuple=True)[0]
            views = _split_consecutive_groups(image_positions)
            if len(views) == 0:
                return None

            num_images = len(views)
            start_view = 1 if (self._include_global and num_images > 1) else 0

            if features.ndim == 3:
                # Hook captured [num_images, tokens_per_view, D] (e.g. Gemma3
                # multi_modal_projector outputs [N, 256, text_dim]).
                # Directly index per-view features.
                for vi in range(start_view, min(features.shape[0], num_images)):
                    tile_feats.append(features[vi])  # [tokens_per_view, D]
                if tile_feats:
                    tokens_per_view = features.shape[1]
                    grid_side = int(tokens_per_view ** 0.5)
                    grid_h_val = grid_w_val = grid_side
            else:
                # 2D [total_tokens, D] — uniform slice per view (InternVL style).
                tokens_per_view = features.shape[0] // num_images
                for vi in range(start_view, num_images):
                    offset_v = vi * tokens_per_view
                    if offset_v + tokens_per_view > features.shape[0]:
                        break
                    tile_feats.append(features[offset_v: offset_v + tokens_per_view])

                grid_side = int(tokens_per_view ** 0.5)
                grid_h_val = grid_w_val = grid_side

        if len(tile_feats) < 2 or grid_h_val is None:
            return None

        stacked = torch.stack(tile_feats)
        return self._densecl_loss(stacked, num_views=len(tile_feats), grid_h=grid_h_val, grid_w=grid_w_val)


class BaselineTrainer:
    """LoRA finetuning driver for commercial VLMs.

    Uses HF Trainer with PEFT LoRA, **not** PyTorch Lightning.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir) / config.model.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API -------------------------------------------------------

    def train(self) -> str:
        """Run LoRA finetuning. Returns path to saved adapter."""
        cfg = self.config
        set_seed(cfg.training.seed)

        # Load model
        model, processor, tokenizer = BaselineModelRegistry.load_model(cfg.model)

        # Determine LoRA target modules
        targets = (
            cfg.lora.target_modules
            or cfg.model.lora_target_modules
            or BaselineModelRegistry.get_default_lora_targets(cfg.model.model_type)
        )

        # Determine task type based on model
        is_seq2seq = cfg.model.model_type.lower() in {"blip2", "blip-2"}
        task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM

        # Apply LoRA
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=targets,
            task_type=task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if cfg.training.gradient_checkpointing:
            model.enable_input_require_grads()

        # Build datasets
        ds_kwargs: Dict[str, Any] = dict(
            processor=processor,
            tokenizer=tokenizer,
            model_type=cfg.model.model_type,
            image_column=cfg.data.image_column,
            instruction_column=cfg.data.instruction_column,
            response_column=cfg.data.response_column,
            pano_view_config=cfg.effective_pano_view,
        )
        train_ds = VLMDataset(
            csv_path=cfg.data_train_csv,
            max_samples=cfg.data.max_train_samples,
            **ds_kwargs,
        )
        pv = cfg.effective_pano_view
        if pv is not None:
            logger.info(
                "Multi-view enabled: strategy=%s hfov=%.0f° overlap=%.1f pitch=[%.0f,%.0f] include_global=%s → multi-image mode",
                pv.strategy, pv.hfov_deg, pv.overlap,
                pv.pitch_min, pv.pitch_max, pv.include_global,
            )
        eval_ds: Optional[VLMDataset] = None
        if cfg.data_val_csv:
            eval_ds = VLMDataset(
                csv_path=cfg.data_val_csv,
                max_samples=cfg.data.max_eval_samples,
                **ds_kwargs,
            )

        # Collate function
        if is_seq2seq:
            collate_fn = _collate_seq2seq(tokenizer)
        else:
            collate_fn = _collate_causal(processor, tokenizer, max_length=cfg.training.max_length)

        # wandb
        report_to: List[str] = []
        if cfg.wandb_enabled:
            report_to.append("wandb")
            if cfg.wandb_project:
                os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

        run_name = f"{cfg.experiment_name}__{cfg.model.name}"

        # Auto batch-size
        batch_size = cfg.training.batch_size
        if batch_size == -1:
            batch_size = autobatch(
                model=model,
                dataset=train_ds,
                collate_fn=collate_fn,
                fraction=0.85,
                default_batch_size=1,
            )
            logger.info("AutoBatch resolved batch_size = %d", batch_size)

        # Training args
        prec = _resolve_precision(cfg.training.mixed_precision)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=cfg.training.num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_ratio=cfg.training.warmup_ratio,
            max_grad_norm=cfg.training.max_grad_norm,
            fp16=prec["fp16"],
            bf16=prec["bf16"],
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            logging_steps=cfg.training.logging_steps,
            eval_strategy=cfg.training.eval_strategy if eval_ds else "no",
            save_strategy=cfg.training.save_strategy,
            save_total_limit=cfg.training.save_total_limit,
            report_to=report_to,
            run_name=run_name,
            seed=cfg.training.seed,
            remove_unused_columns=False,
            dataloader_num_workers=cfg.training.dataloader_num_workers,
            dataloader_pin_memory=cfg.training.dataloader_pin_memory,
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False,
        )

        pa_cfg = cfg.panoadapt
        if pa_cfg is not None and (pa_cfg.spatial_pe or pa_cfg.overlap_loss):
            trainer = _PanoAdaptTrainer(
                panoadapt_config=pa_cfg,
                pano_view_config=cfg.effective_pano_view,
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )
            logger.info(
                "PanoAdapt enabled: spatial_pe=%s overlap_loss=%s (weight=%.3f)",
                pa_cfg.spatial_pe, pa_cfg.overlap_loss, pa_cfg.overlap_loss_weight,
            )
        else:
            trainer = _SafeTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )

        trainer.train()

        # Save LoRA adapter
        save_path = str(self.output_dir / "lora_adapter")
        model.save_pretrained(save_path)
        trainer.save_model(str(self.output_dir / "final"))

        logger.info("Training complete. Adapter saved to %s", save_path)

        # Cleanup
        if isinstance(trainer, _PanoAdaptTrainer):
            trainer.cleanup()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return save_path

    def evaluate(
        self,
        test_csv: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a trained model on *test_csv*. Returns metrics dict.

        Loads the LoRA adapter from ``self.output_dir / "lora_adapter"``
        and runs generation over the test set, computing text metrics.
        """
        cfg = self.config
        eval_dir = Path(output_dir) if output_dir else self.output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        model, processor, tokenizer = BaselineModelRegistry.load_model(cfg.model)

        # Load LoRA adapter
        adapter_path = self.output_dir / "lora_adapter"
        if adapter_path.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            logger.info("Loaded LoRA adapter from %s", adapter_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Read test data
        df = pd.read_csv(test_csv)
        image_paths: List[str] = []
        queries: List[str] = []
        predictions: List[str] = []
        references: List[str] = []

        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
        }
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        tokenizer.padding_side = "left"

        is_seq2seq = cfg.model.model_type.lower() in {"blip2", "blip-2"}

        total = len(df)
        skipped = 0

        with torch.inference_mode():
            pbar = tqdm(range(total), desc="Evaluating", unit="sample", dynamic_ncols=True)
            for i in pbar:
                row = df.iloc[i]
                img_path = row.get(cfg.data.image_column)
                if img_path is None or not Path(str(img_path)).is_file():
                    skipped += 1
                    pbar.set_postfix(skipped=skipped)
                    continue

                erp_image = Image.open(str(img_path)).convert("RGB")
                prompt = str(row.get(cfg.data.instruction_column, "Describe the image."))
                reference = str(row.get(cfg.data.response_column, ""))

                pv = cfg.effective_pano_view
                if is_seq2seq:
                    inputs = processor(images=erp_image, text=prompt, return_tensors="pt")
                elif pv is not None:
                    from cora.processors.anyres_e2p import build_anyres_from_erp
                    import numpy as np

                    # Dispatch strategy-specific params
                    if pv.strategy.lower() == "cubemap":
                        hfov, overlap = 90.0, 0.0
                    else:
                        hfov, overlap = pv.hfov_deg, pv.overlap

                    pack = build_anyres_from_erp(
                        erp_img=erp_image,
                        base_size=pv.base_size,
                        tile_render_size=pv.tile_render_size,
                        vit_size=pv.vit_size,
                        hfov_deg=hfov,
                        overlap=overlap,
                        closed_loop_yaw=pv.closed_loop_yaw,
                        pitch_min=pv.pitch_min,
                        pitch_max=pv.pitch_max,
                    )

                    def _t2pil(t: torch.Tensor) -> Image.Image:
                        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        return Image.fromarray(arr)

                    eval_images: List[Image.Image] = []
                    if pv.include_global:
                        eval_images.append(_t2pil(pack.global_image))
                    for ti in range(pack.tiles.size(0)):
                        eval_images.append(_t2pil(pack.tiles[ti]))

                    image_entries: List[Dict[str, str]] = [
                        {"type": "image"} for _ in eval_images
                    ]
                    messages = [
                        {
                            "role": "user",
                            "content": image_entries + [{"type": "text", "text": prompt}],
                        },
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    inputs = processor(
                        text=[text], images=eval_images, return_tensors="pt", padding=True,
                    )
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    inputs = processor(
                        text=[text], images=[erp_image], return_tensors="pt", padding=True,
                    )

                target_dtype = next(model.parameters()).dtype
                inputs = {
                    k: (
                        v.to(device=device, dtype=target_dtype)
                        if isinstance(v, torch.Tensor) and v.is_floating_point()
                        else v.to(device) if isinstance(v, torch.Tensor) else v
                    )
                    for k, v in inputs.items()
                }

                prompt_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
                outputs = model.generate(**inputs, **gen_kwargs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                out_len = outputs[0].shape[0]

                # Decode full output first, then strip prompt text
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                if is_seq2seq or out_len <= prompt_len:
                    # Seq2seq models or models that return only new tokens
                    pred = full_text
                else:
                    # Strip prompt tokens, then decode
                    generated = outputs[0][prompt_len:]
                    pred = tokenizer.decode(generated, skip_special_tokens=True).strip()

                # Fallback: if sliced decode is empty but full decode has content.
                # - seq2seq/new-tokens-only models: full_text IS the response → use directly.
                # - causal models: full_text includes the prompt; try to extract only the
                #   model/assistant turn by splitting on the role separator.
                if not pred and full_text:
                    if is_seq2seq or out_len <= prompt_len:
                        # full_text is just the new tokens (seq2seq or new-token-only causal)
                        pred = full_text
                    else:
                        # Causal model: full_text = prompt_text + response_text (special tokens stripped).
                        # Try common chat-template model-turn separators to extract just the response.
                        _turn_markers = [
                            "\nmodel\n",       # Gemma3
                            "\nassistant\n",   # generic
                            "\nASSISTANT\n",  # LLaVA-style
                            "\n[/INST]\n",    # Llama-2
                            "\n[/INST]",      # Llama-2 variant
                        ]
                        for _marker in _turn_markers:
                            if _marker in full_text:
                                pred = full_text.split(_marker, 1)[-1].strip()
                                break
                        # If no marker matched, keep pred empty rather than
                        # returning the prompt as the prediction.

                # Log first few samples for debugging
                if i < 3:
                    logger.info(
                        "  [DEBUG] sample=%d prompt_len=%d out_len=%d pred_len=%d pred=%r",
                        i, prompt_len, out_len, len(pred), pred[:80],
                    )

                image_paths.append(str(img_path))
                queries.append(prompt)
                predictions.append(pred)
                references.append(reference)

                # Update progress bar with latest prediction info
                pred_short = pred[:40] + "..." if len(pred) > 40 else pred
                pbar.set_postfix(
                    done=len(predictions),
                    skipped=skipped,
                    pred=pred_short,
                )

        # Compute metrics
        metrics = _compute_basic_metrics(predictions, references)

        try:
            from cora.evaluation.metrics import CORAEvaluator
            evaluator = CORAEvaluator()
            coco_metrics = evaluator.evaluate(predictions, references)
            if coco_metrics:
                metrics.update(coco_metrics)
                CORAEvaluator.print_summary(coco_metrics)
        except Exception as exc:
            logger.warning("COCO metrics failed: %s", exc)

        # Save predictions (JSON)
        records = [
            {"image_path": ip, "query": q, "prediction": p, "reference": r}
            for ip, q, p, r in zip(image_paths, queries, predictions, references)
        ]
        pred_path = eval_dir / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # Save predictions (CSV)
        pred_csv_path = eval_dir / "predictions.csv"
        pd.DataFrame(records).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation complete. Predictions: %s  Metrics: %s", pred_csv_path, metrics_path)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics


# ---------------------------------------------------------------------------
# Basic text metrics (no heavy optional deps required)
# ---------------------------------------------------------------------------


def _compute_basic_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """Compute lightweight text metrics without requiring nltk / rouge-score."""
    import numpy as np

    paired = [
        (p.strip(), r.strip())
        for p, r in zip(predictions, references)
        if r is not None and str(r).strip()
    ]
    if not paired:
        return {"samples": 0}

    preds = [p for p, _ in paired]
    refs = [r for _, r in paired]

    metrics: Dict[str, Any] = {
        "samples": len(paired),
        "exact_match": sum(1 for p, r in paired if p == r) / len(paired),
        "avg_pred_tokens": float(np.mean([len(p.split()) for p in preds])),
        "avg_ref_tokens": float(np.mean([len(r.split()) for r in refs])),
    }

    # Optional: BLEU-4
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

        smooth = SmoothingFunction().method1
        ref_tok = [[r.split()] for r in refs]
        pred_tok = [p.split() for p in preds]
        metrics["bleu4"] = float(
            corpus_bleu(ref_tok, pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        )
    except Exception:
        pass

    # Optional: ROUGE-L
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds) if r and p]
        if scores:
            metrics["rougeL"] = float(np.mean(scores))
    except Exception:
        pass

    return metrics
