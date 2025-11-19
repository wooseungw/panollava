#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HF VLM LoRA Ablation Runner

이 스크립트는 Hugging Face에 등록된 다양한 비전-언어 모델(VLM)에 대해
LoRA 조합을 반복적으로 학습하는 간단한 실험 드라이버입니다.
PanoramaVLM 래퍼에 의존하지 않고, 각 모델의 공식 HF 래퍼를 그대로 로드합니다.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Force single GPU for Qwen2.5-VL (DataParallel incompatible with dynamic resolution)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.utils.data
from PIL import Image

# Allow large panorama images without PIL decompression bomb warnings.
Image.MAX_IMAGE_PIXELS = None  # noqa: E305

import pandas as pd
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from transformers.utils import is_torch_bf16_gpu_available as _is_bf16_supported  # type: ignore
except ImportError:  # pragma: no cover - transformers compatibility shim
    try:
        from transformers.utils import is_bfloat16_supported as _is_bf16_supported  # type: ignore
    except ImportError:  # pragma: no cover - transformers compatibility shim
        def _is_bf16_supported() -> bool:
            if not torch.cuda.is_available():
                return False
            return getattr(torch.cuda, "is_bf16_supported", lambda: False)()

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("PyYAML이 필요합니다. `pip install pyyaml` 로 설치하세요.") from exc

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed


from peft import LoraConfig, TaskType, get_peft_model

try:  # pragma: no cover - optional dependency on latest transformers
    from transformers import LlavaOnevisionForConditionalGeneration
except ImportError:  # pragma: no cover
    LlavaOnevisionForConditionalGeneration = None

try:  # pragma: no cover - optional dependency on latest transformers
    from transformers import Gemma3ForConditionalGeneration
except ImportError:  # pragma: no cover
    Gemma3ForConditionalGeneration = None

# ────────────────────────────────
# 데이터클래스 정의
# ────────────────────────────────


@dataclass
class DataConfig:
    train_csv: str
    val_csv: Optional[str] = None
    image_column: str = "url"
    instruction_column: str = "instruction"
    response_column: str = "response"
    num_workers: int = 4
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class ModelConfig:
    name: str
    hf_model_id: str
    processor_id: Optional[str] = None
    model_type: str = "llava"  # llava | llava_onevision | blip2 | qwen_vl | gemma3
    dtype: str = "float16"
    lora_target_modules: Optional[List[str]] = None


@dataclass
class LoRAVariant:
    name: str
    r: int
    alpha: int
    dropout: float
    target_modules: Optional[List[str]] = None


@dataclass
class TrainingConfig:
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    eval_strategy: str = "no"  # "no" | "steps" | "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    mixed_precision: Optional[str] = None  # "fp16" | "bf16" | None
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    report_to: List[str] = field(default_factory=list)
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = True
    generation_max_new_tokens: int = 128
    generation_min_new_tokens: int = 0
    generation_num_beams: int = 1
    generation_do_sample: bool = False
    generation_temperature: Optional[float] = None
    generation_top_p: Optional[float] = None
    generation_top_k: Optional[int] = None
    generation_repetition_penalty: Optional[float] = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AblationConfig:
    experiment_name: str
    output_dir: str
    data: DataConfig
    models: List[ModelConfig]
    lora_variants: List[LoRAVariant]
    training: TrainingConfig
    wandb: Optional[WandbConfig] = None


# ────────────────────────────────
# Adapter 유틸리티
# ────────────────────────────────


def _collate_with_tokenizer(tokenizer: AutoTokenizer, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not features:
        raise ValueError("collate_fn called with empty batch")

    # Handle variable-sized pixel_values (e.g., Qwen2.5-VL with dynamic resolution)
    pixel_values_list = [f["pixel_values"] for f in features]
    if len(pixel_values_list) > 1:
        shapes = [pv.shape for pv in pixel_values_list]
        if not all(s == shapes[0] for s in shapes):
            # Pad to max length in first dimension
            max_len = max(pv.shape[0] for pv in pixel_values_list)
            hidden_dim = pixel_values_list[0].shape[1]
            padded_pixel_values = []
            for pv in pixel_values_list:
                if pv.shape[0] < max_len:
                    padding = torch.zeros(max_len - pv.shape[0], hidden_dim, dtype=pv.dtype, device=pv.device)
                    pv = torch.cat([pv, padding], dim=0)
                padded_pixel_values.append(pv)
            pixel_values = torch.stack(padded_pixel_values)
        else:
            pixel_values = torch.stack(pixel_values_list)
    else:
        pixel_values = torch.stack(pixel_values_list)

    input_ids = [f["input_ids"] for f in features]
    attention_mask_list = None
    if features[0].get("attention_mask") is not None:
        attention_mask_list = [f["attention_mask"] for f in features]

    pad_inputs_dict = {"input_ids": input_ids}
    if attention_mask_list is not None:
        pad_inputs_dict["attention_mask"] = attention_mask_list

    padded_inputs = tokenizer.pad(
        pad_inputs_dict,
        padding=True,
        return_tensors="pt",
    )

    labels = tokenizer.pad(
        {"input_ids": [f["labels"] for f in features]},
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    batch = {
        "pixel_values": pixel_values,
        "input_ids": padded_inputs["input_ids"],
        "labels": labels,
    }
    if "attention_mask" in padded_inputs:
        batch["attention_mask"] = padded_inputs["attention_mask"]

    # Add any other tensor fields from features (e.g., image_grid_thw, pixel_attention_mask, etc.)
    for key in features[0].keys():
        if key not in batch and key not in ["input_ids", "attention_mask", "labels", "pixel_values"]:
            values = [f[key] for f in features if key in f]
            if values and isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    # If stacking fails due to different shapes, skip this field
                    pass

    return batch


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_str.lower()
    if key not in lookup:
        raise ValueError(f"지원하지 않는 dtype: {dtype_str}")
    if lookup[key] is torch.bfloat16 and not _is_bf16_supported():
        logging.warning("bfloat16이 지원되지 않아 float16으로 대체합니다.")
        return torch.float16
    return lookup[key]


class BaseAdapter:
    """모델 타입별 공통 인터페이스"""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = self._load_processor()
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"Processor {type(self.processor)} 가 tokenizer 를 제공하지 않습니다.")

        if self.tokenizer.pad_token is None and getattr(self.tokenizer, "eos_token", None):
            # Ensure Trainer can pad batches even if tokenizer lacks an explicit pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[assignment]

        # Set padding_side to left for decoder-only models during generation
        self.tokenizer.padding_side = "left"

    # public API ------------------------------------------------------------
    def load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def prepare_example(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function - override in subclasses if needed"""
        return _collate_with_tokenizer(self.tokenizer, features)

    # helper ----------------------------------------------------------------
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        return AutoProcessor.from_pretrained(processor_id)

    def _get_prompt_text(self, row: pd.Series) -> str:
        prompt = str(row.get(self.data_cfg.instruction_column, ""))
        if not prompt or prompt.lower() == "nan":
            prompt = "Describe the image."
        return prompt

    def _get_response_text(self, row: pd.Series) -> str:
        answer = str(row.get(self.data_cfg.response_column, ""))
        if not answer or answer.lower() == "nan":
            answer = ""
        return answer

    def _load_image(self, row: pd.Series) -> Image.Image:
        image_path = row.get(self.data_cfg.image_column)
        if image_path is None:
            raise ValueError(f"CSV에 '{self.data_cfg.image_column}' 열이 없습니다.")
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        return Image.open(img_path).convert("RGB")

    # default target modules
    def default_target_modules(self) -> List[str]:
        if self.model_cfg.lora_target_modules:
            return self.model_cfg.lora_target_modules
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def task_type(self) -> TaskType:
        return TaskType.CAUSAL_LM


class CausalAdapter(BaseAdapter):
    """대화형 VLM (LLaVA, Qwen-VL 등)"""

    @staticmethod
    def _flatten_tensor(value: Any) -> torch.Tensor:
        if value is None:
            raise ValueError("Tensor value is None")
        if isinstance(value, torch.Tensor):
            return value.squeeze(0)
        return torch.as_tensor(value).squeeze(0)

    def prepare_example(self, row: pd.Series) -> Dict[str, Any]:
        """Prepare a single example - keep image as PIL, generate text template only"""
        image = self._load_image(row)
        prompt = self._get_prompt_text(row)
        response = self._get_response_text(row)

        # Format messages
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

        # Generate text templates (don't process images yet - done in collate)
        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "full_text": full_text,
            "prompt_text": prompt_text,
            "image": image,  # Keep as PIL Image
            "reference_text": response,
            "instruction_text": prompt,
            "image_path": str(row.get(self.data_cfg.image_column, "")),
        }

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Batch-level processing with processor"""
        # Extract texts and images
        full_texts = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]
        images = [f["image"] for f in features]

        # DEBUG: Log batch info once
        import sys
        if not hasattr(sys, '_collate_debug_printed'):
            print(f"DEBUG collate_fn: batch_size={len(features)}, num_images={len(images)}")
            sys._collate_debug_printed = True

        # Process batch with processor (handles padding and image processing)
        full_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # DEBUG: Log shapes once
        if not hasattr(sys, '_pixel_shape_printed'):
            if "pixel_values" in full_inputs:
                print(f"DEBUG pixel_values shape: {full_inputs['pixel_values'].shape}")
                if "image_grid_thw" in full_inputs:
                    print(f"DEBUG image_grid_thw: {full_inputs['image_grid_thw']}")
            sys._pixel_shape_printed = True

        # Prepare labels by masking prompt tokens
        labels = full_inputs["input_ids"].clone()

        # For each sample, process prompt separately to find exact masking length
        for i in range(len(features)):
            # Process this sample's prompt individually (not in batch)
            single_prompt_inputs = self.processor(
                text=[prompt_texts[i]],
                images=[images[i]],
                return_tensors="pt",
                padding=False,  # No padding for single sample
            )

            # The actual prompt length without padding
            prompt_len = single_prompt_inputs["input_ids"].shape[1]

            # Mask the prompt tokens in labels
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # Debug: Print label statistics once
        if not hasattr(sys, '_label_stats_printed'):
            for i in range(min(2, len(features))):  # Show first 2 samples
                valid_labels = (labels[i] != -100).sum().item()
                total_tokens = labels[i].shape[0]
                print(f"DEBUG Sample {i}: {valid_labels}/{total_tokens} tokens will be used for loss")
                # Show a snippet of what's being trained on
                response_tokens = labels[i][labels[i] != -100][:20]  # First 20 response tokens
                print(f"DEBUG Sample {i} response tokens (first 20): {response_tokens.tolist()}")
            sys._label_stats_printed = True

        # Build result batch
        result = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }

        # Add image-related tensors (pixel_values, image_grid_thw, etc.)
        for key in full_inputs.keys():
            if key not in result:
                result[key] = full_inputs[key]

        return result

class LlavaAdapter(CausalAdapter):
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        return LlavaProcessor.from_pretrained(processor_id)

    def load_model(self) -> torch.nn.Module:
        dtype = _resolve_dtype(self.model_cfg.dtype)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


class QwenVLAdapter(CausalAdapter):
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        processor = AutoProcessor.from_pretrained(processor_id)
        # Override image processor to use fixed 224x224 resolution
        processor.image_processor.min_pixels = 224 * 224
        processor.image_processor.max_pixels = 224 * 224
        return processor

    def load_model(self) -> torch.nn.Module:
        dtype = _resolve_dtype(self.model_cfg.dtype)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


class LlavaOneVisionAdapter(CausalAdapter):
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        return AutoProcessor.from_pretrained(processor_id)

    def load_model(self) -> torch.nn.Module:
        if LlavaOnevisionForConditionalGeneration is None:
            raise ImportError(
                "transformers 패키지에 LlavaOnevisionForConditionalGeneration 이 없습니다. "
                "pip install --upgrade transformers 로 업데이트하세요."
            )
        dtype = _resolve_dtype(self.model_cfg.dtype)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


class Gemma3Adapter(CausalAdapter):
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        return AutoProcessor.from_pretrained(processor_id)

    def load_model(self) -> torch.nn.Module:
        if Gemma3ForConditionalGeneration is None:
            raise ImportError(
                "transformers 패키지에 Gemma3ForConditionalGeneration 이 없습니다. "
                "pip install --upgrade transformers 로 업데이트하세요."
            )
        dtype = _resolve_dtype(self.model_cfg.dtype)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


class Seq2SeqAdapter(BaseAdapter):
    """BLIP-2 같이 인코더-디코더 구조"""

    def task_type(self) -> TaskType:
        return TaskType.SEQ_2_SEQ_LM

    def prepare_example(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        image = self._load_image(row)
        prompt = self._get_prompt_text(row)
        response = self._get_response_text(row)

        # BLIP-2: Following HuggingFace official example, use same text for input_ids and labels
        # The model internally handles the shift for language modeling
        text = response  # Use the target caption

        enc_inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        # Labels = input_ids (model handles the shift internally)
        labels = enc_inputs["input_ids"].squeeze(0).clone()

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # Debug
        import sys
        if not hasattr(sys, '_labels_debug_printed'):
            valid_labels = (labels != -100).sum().item()
            print(f"DEBUG prepare_example (BLIP-2):")
            print(f"  text: '{text[:50] if len(text) > 50 else text}'")
            print(f"  input_ids==labels length={len(labels)}, valid_tokens={valid_labels}")
            sys._labels_debug_printed = True

        result = {
            "labels": labels,
        }

        # Add all processor outputs with squeeze(0)
        for key, value in enc_inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)
            elif value is not None:
                result[key] = value

        result["image"] = image
        result["prompt_text"] = prompt
        result["reference_text"] = response
        result["instruction_text"] = prompt
        result["image_path"] = str(row.get(self.data_cfg.image_column, ""))

        return result

    def prepare_for_generation(self, rows, processor):
        """Prepare batch for generation - BLIP-2 specific handling"""
        texts = [self._get_prompt_text(row) for row in rows]
        images = [self._load_image(row) for row in rows]

        # BLIP-2 generation only needs pixel_values (no input_ids required)
        processed = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        return processed

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """BLIP-2 specific collate function for seq2seq training"""
        # Stack pixel_values (images)
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        # Pad input_ids (prompts for encoder)
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f.get("attention_mask") for f in features if f.get("attention_mask") is not None]

        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list} if attention_mask_list else {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt",
        )

        # Pad labels (target responses)
        labels = self.tokenizer.pad(
            {"input_ids": [f["labels"] for f in features]},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # Debug: Print shapes once
        import sys
        if not hasattr(sys, '_collate_shapes_printed'):
            print(f"DEBUG collate_fn shapes:")
            print(f"  pixel_values: {pixel_values.shape}")
            print(f"  input_ids: {padded_inputs['input_ids'].shape}")
            print(f"  labels: {labels.shape}")
            print(f"  Sample input_ids[0]: {padded_inputs['input_ids'][0][:10].tolist()}")
            print(f"  Sample labels[0]: {labels[0][:10].tolist()}")
            sys._collate_shapes_printed = True

        batch = {
            "pixel_values": pixel_values,
            "input_ids": padded_inputs["input_ids"],
            "labels": labels,
        }

        if "attention_mask" in padded_inputs:
            batch["attention_mask"] = padded_inputs["attention_mask"]

        return batch

class Blip2Adapter(Seq2SeqAdapter):
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        return Blip2Processor.from_pretrained(processor_id)

    def load_model(self) -> torch.nn.Module:
        dtype = _resolve_dtype(self.model_cfg.dtype)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


def build_adapter(model_cfg: ModelConfig, data_cfg: DataConfig) -> BaseAdapter:
    model_type = model_cfg.model_type.lower()
    if model_type in {"llava", "llava_next", "llava_gemma"}:
        return LlavaAdapter(model_cfg, data_cfg)
    if model_type in {"llava_onevision", "llava-onevision", "onevision"}:
        return LlavaOneVisionAdapter(model_cfg, data_cfg)
    if model_type in {"qwen_vl", "qwenvl", "qwen2_vl"}:
        return QwenVLAdapter(model_cfg, data_cfg)
    if model_type in {"gemma3", "gemma-3"}:
        return Gemma3Adapter(model_cfg, data_cfg)
    if model_type in {"blip2", "blip-2"}:
        return Blip2Adapter(model_cfg, data_cfg)
    raise ValueError(f"지원하지 않는 model_type: {model_cfg.model_type}")


# ────────────────────────────────
# Dataset & Ablation Runner
# ────────────────────────────────


class VLMTrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, adapter: BaseAdapter, max_samples: Optional[int] = None):
        self.df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples > 0:
            self.df = self.df.head(max_samples)
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        return self.adapter.prepare_example(row)


class LoRAAblationRunner:
    def __init__(self, config: AblationConfig, override_output_dir: Optional[Path] = None, override_exp_name: Optional[str] = None):
        self.config = config
        if override_output_dir is not None:
            config.output_dir = str(override_output_dir)
        if override_exp_name is not None:
            config.experiment_name = override_exp_name

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("vlm_ablation")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        self.wandb_cfg = config.wandb or WandbConfig()
        self._wandb_enabled = bool(self.wandb_cfg.enabled)
        self._wandb_base_tags: List[str] = []
        if self._wandb_enabled:
            self._init_wandb_env()

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value)

    def _init_wandb_env(self) -> None:
        project = self.wandb_cfg.project or self.config.experiment_name
        os.environ.setdefault("WANDB_PROJECT", project)
        if self.wandb_cfg.entity:
            os.environ.setdefault("WANDB_ENTITY", self.wandb_cfg.entity)
        group_template = self.wandb_cfg.group or "{experiment}"
        group_value = group_template.format(experiment=self.config.experiment_name)
        os.environ.setdefault("WANDB_RUN_GROUP", group_value)
        if self.wandb_cfg.mode:
            os.environ.setdefault("WANDB_MODE", self.wandb_cfg.mode)
        self._wandb_base_tags = list(dict.fromkeys(self.wandb_cfg.tags or []))

    def _build_wandb_run_name(self, model_cfg: ModelConfig, lora_cfg: LoRAVariant) -> str:
        return f"{self.config.experiment_name}__{model_cfg.name}__{lora_cfg.name}"

    def _build_wandb_tags(self, model_cfg: ModelConfig, lora_cfg: LoRAVariant) -> List[str]:
        tags = self._wandb_base_tags + [
            self.config.experiment_name,
            model_cfg.name,
            lora_cfg.name,
        ]
        return list(dict.fromkeys(tag for tag in tags if tag))

    def _prepare_wandb_run(self, run_name: str, tags: List[str]) -> None:
        if not self._wandb_enabled:
            return
        if tags:
            os.environ["WANDB_TAGS"] = ",".join(tags)
        os.environ["WANDB_NAME"] = run_name
        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        set_seed(self.config.training.seed)
        all_results: List[Dict[str, Any]] = []

        for model_cfg in self.config.models:
            adapter = build_adapter(model_cfg, self.config.data)
            self.logger.info(f"모델 준비 완료: {model_cfg.name} ({model_cfg.hf_model_id})")

            train_dataset = self._build_dataset(
                csv_path=self.config.data.train_csv,
                adapter=adapter,
                max_samples=self.config.data.max_train_samples,
                required=True,
            )
            assert train_dataset is not None  # required=True guarantees dataset 생성
            eval_dataset = self._build_dataset(
                csv_path=self.config.data.val_csv,
                adapter=adapter,
                max_samples=self.config.data.max_eval_samples,
                required=False,
            )

            for lora_cfg in self.config.lora_variants:
                exp_id = f"{model_cfg.name}__{lora_cfg.name}"
                self.logger.info(f"=== 실험 시작: {exp_id} ===")

                model = self._load_fresh_model(adapter)

                target_modules = (
                    lora_cfg.target_modules
                    or model_cfg.lora_target_modules
                    or adapter.default_target_modules()
                )

                peft_config = LoraConfig(
                    r=lora_cfg.r,
                    lora_alpha=lora_cfg.alpha,
                    lora_dropout=lora_cfg.dropout,
                    target_modules=target_modules,
                    task_type=adapter.task_type(),
                )

                lora_model = get_peft_model(model, peft_config)
                lora_model.print_trainable_parameters()

                exp_dir = self.output_dir / exp_id
                exp_dir.mkdir(parents=True, exist_ok=True)

                run_name = self._build_wandb_run_name(model_cfg, lora_cfg)
                wandb_tags = self._build_wandb_tags(model_cfg, lora_cfg)
                self._prepare_wandb_run(run_name, wandb_tags)

                training_args = self._build_training_args(exp_dir, run_name=run_name)
                trainer = Trainer(
                    model=lora_model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=adapter.tokenizer,
                    data_collator=adapter.collate_fn,
                )

                train_metrics = trainer.train()
                trainer.save_model(str(exp_dir / "final"))

                metrics = train_metrics.metrics
                if eval_dataset is not None and training_args.eval_strategy != "no":
                    eval_metrics = trainer.evaluate()
                    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

                if eval_dataset is not None:
                    gen_metrics = self._run_generation_evaluation(
                        model=trainer.model,
                        adapter=adapter,
                        dataset=eval_dataset,
                        exp_dir=exp_dir,
                        experiment_id=exp_id,
                    )
                    if gen_metrics:
                        metrics.update({f"gen_{k}": v for k, v in gen_metrics.items()})

                metrics["experiment_id"] = exp_id
                metrics["model"] = model_cfg.name
                metrics["lora_variant"] = lora_cfg.name

                metrics_path = exp_dir / "metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                self.logger.info(f"실험 완료: {exp_id} -> {metrics_path}")
                all_results.append(metrics)

                # LoRA 가중치만 저장
                lora_model.save_pretrained(str(exp_dir / "lora_adapter"))

                del trainer
                del lora_model
                del model
                gc.collect()
                if torch.cuda.is_available():  # pragma: no cover - GPU only
                    torch.cuda.empty_cache()

        summary_path = self.output_dir / "ablation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"요약 저장: {summary_path}")
        return all_results

    # ------------------------------------------------------------------
    def _load_fresh_model(self, adapter: BaseAdapter) -> torch.nn.Module:
        model = adapter.load_model()
        config = getattr(model, "config", None)
        if config is not None:
            setattr(config, "use_cache", False)
        return model

    def _build_dataset(
        self,
        *,
        csv_path: Optional[str],
        adapter: BaseAdapter,
        max_samples: Optional[int],
        required: bool,
    ) -> Optional[VLMTrainDataset]:
        if not csv_path:
            if required:
                raise ValueError("train_csv가 설정되지 않았습니다.")
            return None

        csv_file = Path(csv_path)
        if not csv_file.is_file():
            message = f"CSV 파일을 찾을 수 없습니다: {csv_file}"
            if required:
                raise FileNotFoundError(message)
            self.logger.warning(message)
            return None

        return VLMTrainDataset(csv_path=csv_path, adapter=adapter, max_samples=max_samples)

    def _resolve_precision_flags(self, mixed_precision: Optional[str]) -> Dict[str, bool]:
        if mixed_precision is None:
            return {"fp16": False, "bf16": False}

        key = mixed_precision.lower()
        if key == "bf16":
            if _is_bf16_supported():
                return {"fp16": False, "bf16": True}
            self.logger.warning("bf16을 지원하지 않는 환경입니다. fp16으로 대체합니다.")
            return {"fp16": True, "bf16": False}
        if key == "fp16":
            return {"fp16": True, "bf16": False}

        self.logger.warning(f"알 수 없는 mixed_precision 값 '{mixed_precision}' → 자동으로 해제합니다.")
        return {"fp16": False, "bf16": False}

    # ------------------------------------------------------------------
    def _build_training_args(self, exp_dir: Path, *, run_name: str) -> TrainingArguments:
        cfg = self.config.training
        precision_flags = self._resolve_precision_flags(cfg.mixed_precision)
        report_to = list(cfg.report_to or [])
        if self._wandb_enabled and "wandb" not in report_to:
            report_to.append("wandb")

        return TrainingArguments(
            output_dir=str(exp_dir / "checkpoints"),
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            save_total_limit=cfg.save_total_limit,
            fp16=precision_flags["fp16"],
            bf16=precision_flags["bf16"],
            gradient_checkpointing=cfg.gradient_checkpointing,
            max_grad_norm=cfg.max_grad_norm,
            report_to=report_to,
            run_name=run_name,
            dataloader_pin_memory=cfg.dataloader_pin_memory,
            dataloader_num_workers=self.config.data.num_workers,
            dataloader_persistent_workers=cfg.dataloader_persistent_workers,
            seed=cfg.seed,
            remove_unused_columns=False,
            # Disable DataParallel for Qwen2.5-VL (incompatible with dynamic resolution batching)
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False,
        )

    def _prepare_generation_kwargs(self) -> Dict[str, Any]:
        cfg = self.config.training
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(cfg.generation_max_new_tokens, 1),
        }
        if cfg.generation_min_new_tokens and cfg.generation_min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = cfg.generation_min_new_tokens
        if cfg.generation_num_beams and cfg.generation_num_beams > 1:
            gen_kwargs["num_beams"] = cfg.generation_num_beams
        if cfg.generation_do_sample:
            gen_kwargs["do_sample"] = True
        if cfg.generation_temperature is not None:
            gen_kwargs["temperature"] = cfg.generation_temperature
        if cfg.generation_top_p is not None:
            gen_kwargs["top_p"] = cfg.generation_top_p
        if cfg.generation_top_k is not None:
            gen_kwargs["top_k"] = cfg.generation_top_k
        if cfg.generation_repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = cfg.generation_repetition_penalty
        return gen_kwargs

    def _write_jsonl(self, path: Path, records: Iterable[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")

    def _compute_text_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        paired = [
            (pred.strip(), ref.strip())
            for pred, ref in zip(predictions, references)
            if ref is not None and str(ref).strip() != ""
        ]

        if not paired:
            self.logger.warning("평가 가능한 예측-정답 쌍이 없어 텍스트 메트릭 계산을 건너뜁니다.")
            return metrics

        preds = [p for p, _ in paired]
        refs = [r for _, r in paired]

        metrics["samples"] = float(len(paired))
        metrics["exact_match"] = float(sum(1 for p, r in paired if p == r) / len(paired))
        metrics["avg_pred_tokens"] = float(np.mean([len(p.split()) for p in preds]) if preds else 0.0)
        metrics["avg_ref_tokens"] = float(np.mean([len(r.split()) for r in refs]) if refs else 0.0)

        # BLEU-4
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

            smoothing = SmoothingFunction().method1
            ref_tokens = [[r.split()] for r in refs]
            pred_tokens = [p.split() for p in preds]
            if ref_tokens and pred_tokens:
                metrics["bleu4"] = float(corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.warning(f"BLEU-4 계산을 건너뜁니다: {exc}")

        # METEOR
        try:
            import nltk

            try:  # Ensure required corpora 존재
                nltk.data.find('corpora/wordnet')
            except LookupError:  # pragma: no cover - optional download
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)

            from nltk.translate.meteor_score import meteor_score

            meteor_scores = []
            for ref, pred in zip(refs, preds):
                if ref and pred:
                    meteor_scores.append(meteor_score([ref.split()], pred.split()))
            if meteor_scores:
                metrics["meteor"] = float(np.mean(meteor_scores))
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.warning(f"METEOR 계산을 건너뜁니다: {exc}")

        # ROUGE-L
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_values = []
            for ref, pred in zip(refs, preds):
                if ref and pred:
                    rouge = scorer.score(ref, pred)
                    rouge_values.append(rouge['rougeL'].fmeasure)
            if rouge_values:
                metrics["rougeL"] = float(np.mean(rouge_values))
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.warning(f"ROUGE-L 계산을 건너뜁니다: {exc}")

        return metrics

    def _run_generation_evaluation(
        self,
        *,
        model: torch.nn.Module,
        adapter: BaseAdapter,
        dataset: VLMTrainDataset,
        exp_dir: Path,
        experiment_id: str,
    ) -> Dict[str, float]:
        if len(dataset) == 0:
            self.logger.warning("검증 데이터셋이 비어 있어 예측 생성을 건너뜁니다.")
            return {}

        gen_kwargs = self._prepare_generation_kwargs()
        tokenizer = adapter.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        if pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", pad_token_id)
        if eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", eos_token_id)

        device = next(model.parameters()).device
        model.eval()

        df = dataset.df.reset_index(drop=True)
        batch_size = max(1, self.config.training.per_device_eval_batch_size)
        predictions: List[str] = []
        references: List[str] = []
        instructions: List[str] = []
        image_paths: List[str] = []
        prompt_texts: List[str] = []

        self.logger.info(f"예측 생성 시작 ({experiment_id}) - 총 {len(df)} 샘플, 배치 {batch_size}")

        batch_iter = tqdm(
            range(0, len(df), batch_size),
            total=math.ceil(len(df) / batch_size) if batch_size else len(df),
            desc="Generating captions",
        )

        with torch.inference_mode():
            for start in batch_iter:
                end = min(start + batch_size, len(df))
                rows = [df.iloc[i] for i in range(start, end)]
                examples = [adapter.prepare_example(row) for row in rows]

                texts = [ex.get("prompt_text") or "" for ex in examples]
                images = [ex.get("image") for ex in examples]
                if any(img is None for img in images):
                    images = [adapter._load_image(row) for row in rows]

                processed = adapter.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )

                input_ids = processed.get("input_ids")
                attention_mask = processed.get("attention_mask")
                if isinstance(input_ids, torch.Tensor):
                    if pad_token_id is not None:
                        prompt_lengths = (input_ids != pad_token_id).sum(dim=1)
                    elif isinstance(attention_mask, torch.Tensor):
                        prompt_lengths = attention_mask.sum(dim=1)
                    else:
                        prompt_lengths = torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long)
                else:
                    prompt_lengths = torch.zeros(len(rows), dtype=torch.long)

                processed_tensors = {
                    key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                    for key, value in processed.items()
                }

                # Debug: Print generation inputs once
                import sys
                if not hasattr(sys, '_gen_debug_printed'):
                    self.logger.info(f"Generation inputs: {list(processed_tensors.keys())}")
                    self.logger.info(f"Generation kwargs: {gen_kwargs}")
                    if "input_ids" in processed_tensors:
                        self.logger.info(f"Input IDs shape: {processed_tensors['input_ids'].shape}")
                    sys._gen_debug_printed = True

                try:
                    outputs = model.generate(**processed_tensors, **gen_kwargs)
                except Exception as e:
                    self.logger.error(f"Generation failed: {e}")
                    self.logger.error(f"Available keys: {list(processed_tensors.keys())}")
                    self.logger.error(f"Generation kwargs: {gen_kwargs}")
                    raise
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                outputs = outputs.detach().cpu()
                prompt_lengths = prompt_lengths.cpu()

                for idx, row in enumerate(rows):
                    gen_ids = outputs[idx].tolist()
                    cut = int(prompt_lengths[idx].item()) if idx < len(prompt_lengths) else 0
                    generated_tokens = gen_ids[cut:]
                    prediction_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    reference_raw = row.get(self.config.data.response_column, "")
                    instruction_raw = row.get(self.config.data.instruction_column, "")
                    image_raw = row.get(self.config.data.image_column, "")

                    predictions.append(prediction_text)
                    references.append(self._safe_str(reference_raw))
                    instructions.append(self._safe_str(instruction_raw))
                    image_paths.append(self._safe_str(image_raw))
                    prompt_texts.append(examples[idx].get("prompt_text", ""))

        records = []
        for idx, (pred, ref, instr, image_path, prompt_text) in enumerate(zip(predictions, references, instructions, image_paths, prompt_texts)):
            records.append(
                {
                    "index": idx,
                    "image_path": image_path,
                    "instruction": instr,
                    "prompt_text": prompt_text,
                    "prediction": pred,
                    "reference": ref,
                }
            )

        predictions_path = exp_dir / "predictions.jsonl"
        self._write_jsonl(predictions_path, records)
        self.logger.info(f"예측 저장: {predictions_path}")

        metrics = self._compute_text_metrics(predictions, references)
        if metrics:
            metrics_path = exp_dir / "generation_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            self.logger.info(f"생성 기반 메트릭 저장: {metrics_path}")

        return metrics


# ────────────────────────────────
# Config 로딩 & CLI
# ────────────────────────────────


def load_config(path: Path) -> AblationConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # YAML에서 숫자 값들을 제대로 변환
    def convert_numeric_values(data):
        if isinstance(data, dict):
            return {k: convert_numeric_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_numeric_values(item) for item in data]
        elif isinstance(data, str):
            # 과학적 표기법 문자열을 float로 변환 시도
            try:
                if 'e' in data.lower() or '.' in data:
                    return float(data)
            except ValueError:
                pass
            return data
        else:
            return data

    raw = convert_numeric_values(raw)

    data_cfg = DataConfig(**raw["data"])
    model_cfgs = [ModelConfig(**item) for item in raw["models"]]
    lora_cfgs = [LoRAVariant(**item) for item in raw["lora_variants"]]
    training_cfg = TrainingConfig(**raw["training"])
    wandb_cfg = WandbConfig(**raw["wandb"]) if raw.get("wandb") else None

    return AblationConfig(
        experiment_name=raw["experiment_name"],
        output_dir=raw["output_dir"],
        data=data_cfg,
        models=model_cfgs,
        lora_variants=lora_cfgs,
        training=training_cfg,
        wandb=wandb_cfg,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF VLM LoRA Ablation Study")
    parser.add_argument("--config", type=str, default="configs/vlm_ablation.yaml", help="설정 파일 경로")
    parser.add_argument("--output-dir", type=str, default=None, help="결과 저장 경로 (옵션)")
    parser.add_argument("--experiment-name", type=str, default=None, help="실험 이름 (옵션)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    override_output = Path(args.output_dir) if args.output_dir else None
    runner = LoRAAblationRunner(config, override_output_dir=override_output, override_exp_name=args.experiment_name)
    runner.run()


if __name__ == "__main__":
    main()
