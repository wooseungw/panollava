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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Force single GPU for Qwen2.5-VL (DataParallel incompatible with dynamic resolution)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.utils.data
from PIL import Image

import pandas as pd

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
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed


from peft import LoraConfig, TaskType, get_peft_model

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
    model_type: str = "llava"  # llava | blip2 | qwen_vl | auto
    torch_dtype: str = "float16"
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


@dataclass
class AblationConfig:
    experiment_name: str
    output_dir: str
    data: DataConfig
    models: List[ModelConfig]
    lora_variants: List[LoRAVariant]
    training: TrainingConfig


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

        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Prepare labels
        labels = full_inputs["input_ids"].clone()

        # Mask prompt tokens for each sample in batch
        for i in range(len(features)):
            prompt_len = (prompt_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum()
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

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
        dtype = _resolve_dtype(self.model_cfg.torch_dtype)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


class QwenVLAdapter(CausalAdapter):
    def load_model(self) -> torch.nn.Module:
        dtype = _resolve_dtype(self.model_cfg.torch_dtype)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        enc_inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        # labels는 tokenizer를 직접 호출
        target = self.processor.tokenizer(
            response,
            return_tensors="pt",
            padding="longest",
        )

        labels = target["input_ids"].squeeze(0)
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "labels": labels,
        }

        # Add all processor outputs with squeeze(0)
        for key, value in enc_inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)
            elif value is not None:
                result[key] = value

        return result

class Blip2Adapter(Seq2SeqAdapter):
    def _load_processor(self):
        processor_id = self.model_cfg.processor_id or self.model_cfg.hf_model_id
        return Blip2Processor.from_pretrained(processor_id)

    def load_model(self) -> torch.nn.Module:
        dtype = _resolve_dtype(self.model_cfg.torch_dtype)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_cfg.hf_model_id,
            torch_dtype=dtype,
        )
        return model


def build_adapter(model_cfg: ModelConfig, data_cfg: DataConfig) -> BaseAdapter:
    model_type = model_cfg.model_type.lower()
    if model_type in {"llava", "llava_next", "llava_gemma"}:
        return LlavaAdapter(model_cfg, data_cfg)
    if model_type in {"qwen_vl", "qwenvl", "qwen2_vl"}:
        return QwenVLAdapter(model_cfg, data_cfg)
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

                training_args = self._build_training_args(exp_dir)
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
    def _build_training_args(self, exp_dir: Path) -> TrainingArguments:
        cfg = self.config.training
        precision_flags = self._resolve_precision_flags(cfg.mixed_precision)

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
            report_to=cfg.report_to,
            dataloader_pin_memory=cfg.dataloader_pin_memory,
            dataloader_num_workers=self.config.data.num_workers,
            dataloader_persistent_workers=cfg.dataloader_persistent_workers,
            seed=cfg.seed,
            remove_unused_columns=False,
            # Disable DataParallel for Qwen2.5-VL (incompatible with dynamic resolution batching)
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False,
        )


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

    return AblationConfig(
        experiment_name=raw["experiment_name"],
        output_dir=raw["output_dir"],
        data=data_cfg,
        models=model_cfgs,
        lora_variants=lora_cfgs,
        training=training_cfg,
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
