"""Baseline LoRA finetuning configuration schema.

Pydantic-based configuration for commercial VLM LoRA finetuning,
ported from legacy/root_scripts/vlm_finetune_and_eval.py dataclasses.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BaselineModelConfig(BaseModel):
    """Configuration for a single commercial VLM."""

    name: str = "qwen2.5-vl-7b"
    hf_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor_id: Optional[str] = None  # defaults to hf_model_id
    model_type: str = "qwen_vl"  # qwen_vl, llava, llava_onevision, blip2, gemma3
    dtype: str = "float16"
    lora_target_modules: Optional[List[str]] = None
    image_size: int = 224
    dynamic_resolution: bool = False
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


class BaselineLoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # overrides model default


class BaselineTrainingConfig(BaseModel):
    """Training hyperparameters for HF Trainer."""

    num_epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = False
    mixed_precision: Optional[str] = None  # fp16, bf16, None
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    eval_strategy: str = "no"
    max_length: Optional[int] = 512  # Max input sequence length (truncation). None=no limit
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = False


class BaselineDataConfig(BaseModel):
    """Column mapping for CSV datasets."""

    image_column: str = "url"
    instruction_column: str = "instruction"
    response_column: str = "response"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class PanoViewConfig(BaseModel):
    """Multi-view panoramic input configuration.

    Strategies:
      - ``"anyres_e2p"``: 1 global resized + N yaw tiles (default: 1+8=9 views)
      - ``"cubemap"``: 4 side faces at 90° FOV (+ optional global = 4 or 5 views)
      - ``"pinhole"``: N yaw tiles only, no global context (default: 8 views)
    """

    strategy: str = "anyres_e2p"
    include_global: bool = True
    hfov_deg: float = 45.0
    overlap: float = 0.0
    closed_loop_yaw: bool = True
    pitch_min: float = 0.0
    pitch_max: float = 0.0
    base_size: int = 336
    tile_render_size: int = 672
    vit_size: Optional[int] = None


# Backward-compatible alias
AnyresE2PConfig = PanoViewConfig


class PanoAdaptConfig(BaseModel):
    """PanoAdapt: spatial PE (Layer 2) + overlap loss (Layer 3) for panoramic VLMs."""

    model_type: str = "qwen_vl"  # qwen_vl, qwen2_vl, internvl, gemma3
    rope_type: str = "3d"  # 3d (Qwen M-RoPE), 1d (InternVL/Gemma3 PanoRoPE-1D)
    spatial_pe: bool = False
    overlap_loss: bool = False
    overlap_loss_type: str = "densecl"  # densecl, vicreg_batchwise, vicreg_pairwise
    overlap_loss_weight: float = 0.0
    overlap_loss_temperature: float = 0.07
    overlap_ratio: float = 0.5
    vicreg_sim_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0


class BaselineConfig(BaseModel):
    """Top-level configuration for baseline LoRA finetuning."""

    experiment_name: str = "baseline_finetune"
    output_dir: str = "runs/baseline"
    model: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    lora: BaselineLoRAConfig = Field(default_factory=BaselineLoRAConfig)
    training: BaselineTrainingConfig = Field(default_factory=BaselineTrainingConfig)
    data: BaselineDataConfig = Field(default_factory=BaselineDataConfig)
    pano_view: Optional[PanoViewConfig] = None
    anyres_e2p: Optional[PanoViewConfig] = None
    panoadapt: Optional[PanoAdaptConfig] = None
    max_new_tokens: int = 128
    data_train_csv: str = "data/quic360/train.csv"
    data_val_csv: Optional[str] = None
    data_test_csv: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_enabled: bool = False

    @property
    def effective_pano_view(self) -> Optional[PanoViewConfig]:
        """Return pano_view config, falling back to anyres_e2p for backward compat."""
        return self.pano_view or self.anyres_e2p
