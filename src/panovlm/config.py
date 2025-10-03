#!/usr/bin/env python3
# coding: utf-8
"""
PanoramaVLM 통합 설정 시스템 v2.0
=================================

Pydantic 기반 타입 안전 설정 관리:
- YAML 우선 (JSON은 legacy 호환만)
- 자동 validation
- 리샘플러별 기본값 자동 적용
- train.py, model.py와 완전 호환
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Union
from typing_extensions import Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
import json
import warnings


# ============================================================================
# 리샘플러 타입별 기본 설정
# ============================================================================

RESAMPLER_CONFIGS = {
    "mlp": {
        "latent_dimension": 768,
        "depth": 3,
        "hidden_dim": 1536,
        "use_ln": True,
    },
    "perceiver": {
        "latent_dimension": 768,
        "num_latents": 32,
        "depth": 4,
        "heads": 8,
        "use_ln": True,
    },
    "bimamba": {
        "latent_dimension": 768,
        "depth": 4,
        "hidden_dim": 1024,
        "use_ln": True,
        "dropout": 0.05,
        "d_state": 64,
        "d_conv": 4,
        "expand": 1.75,
        "norm_first": True,
        "enable_cross_view": False,
    },
    "bidirectional_mamba": {
        "latent_dimension": 768,
        "depth": 4,
        "hidden_dim": 1024,
        "use_ln": True,
        "dropout": 0.05,
        "d_state": 64,
        "d_conv": 4,
        "expand": 1.75,
        "norm_first": True,
        "enable_cross_view": False,
    },
    "bi_mamba": {
        "latent_dimension": 768,
        "depth": 4,
        "hidden_dim": 1024,
        "use_ln": True,
        "dropout": 0.05,
        "d_state": 64,
        "d_conv": 4,
        "expand": 1.75,
        "norm_first": True,
        "enable_cross_view": False,
    },
    "qformer": {
        "latent_dimension": 768,
        "depth": 6,
        "hidden_dim": 768,
        "use_ln": True,
        "dropout": 0.1,
        "num_query_tokens": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
    },
}


# ============================================================================
# Resampler Config Models
# ============================================================================

class ResamplerConfigBase(BaseModel):
    """리샘플러 기본 설정"""
    latent_dimension: int = Field(default=768, description="Latent dimension")
    depth: int = Field(default=2, gt=0, description="Number of resampler layers")
    use_ln: bool = Field(default=True, description="Use LayerNorm in resampler")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")
    num_views: int = Field(default=8, gt=0, description="Number of panorama views")

    class Config:
        extra = "forbid"


class MLPResamplerConfig(ResamplerConfigBase):
    """MLP Resampler 설정"""
    hidden_dim: int = Field(default=1536, gt=0, description="Hidden dimension")


class PerceiverResamplerConfig(ResamplerConfigBase):
    """Perceiver Resampler 설정"""
    num_latents: int = Field(default=32, gt=0, description="Number of latent tokens")
    heads: int = Field(default=8, gt=0, description="Number of attention heads")


class BiMambaResamplerConfig(ResamplerConfigBase):
    """BiMamba/BidirectionalMamba Resampler 설정"""

    hidden_dim: int = Field(default=1536, gt=0, description="Hidden dimension")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout applied inside Mamba blocks")
    d_state: int = Field(default=64, gt=0, description="State size for the Mamba kernel")
    d_conv: int = Field(default=4, gt=0, description="Convolution width for the Mamba kernel")
    expand: float = Field(default=2.0, gt=0.0, description="Expansion ratio in the Mamba feed-forward path")
    norm_first: bool = Field(default=True, description="Apply LayerNorm before Mamba blocks")
    enable_cross_view: bool = Field(default=False, description="Enable cross-view attention")

    @model_validator(mode='before')
    @classmethod
    def _apply_aliases(cls, values):
        if isinstance(values, dict) and 'num_layers' in values:
            values = dict(values)
            values.setdefault('depth', values['num_layers'])
            values.pop('num_layers')
        return values


# --------------------------------------------------------------------------
# QFormer Resampler configuration
# --------------------------------------------------------------------------
class QFormerResamplerConfig(ResamplerConfigBase):
    """QFormer-style Resampler 설정"""

    hidden_dim: int = Field(default=768, gt=0, description="Hidden size inside the QFormer encoder")
    num_query_tokens: int = Field(default=64, gt=0, description="Number of learnable query tokens")
    num_hidden_layers: int = Field(default=6, gt=0, description="Number of transformer layers")
    num_attention_heads: int = Field(default=8, gt=0, description="Attention heads for the QFormer encoder")

    @model_validator(mode='before')
    @classmethod
    def _apply_aliases(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            if 'num_layers' in values and 'num_hidden_layers' not in values:
                values['num_hidden_layers'] = values.pop('num_layers')
            if 'hidden_size' in values and 'hidden_dim' not in values:
                values['hidden_dim'] = values['hidden_size']
        return values


# Union type for all resampler configs
ResamplerConfig = Union[MLPResamplerConfig, PerceiverResamplerConfig, BiMambaResamplerConfig, QFormerResamplerConfig]


# ============================================================================
# Pydantic Config Models
# ============================================================================

class ModelConfig(BaseModel):
    """
    모델 아키텍처 설정

    train.py의 VLMModule, model.py의 PanoramaVLM과 호환
    """

    # 필수 설정 (사용자가 YAML에서 지정)
    vision_name: str = Field(
        default="google/siglip-base-patch16-224",
        description="Vision encoder model name"
    )
    language_model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Language model name"
    )
    resampler_type: Literal["mlp", "perceiver", "bimamba", "bidirectional_mamba", "bi_mamba", "qformer"] = Field(
        default="mlp",
        description="Resampler architecture type"
    )

    # 리샘플러 세부 설정 (resampler_type에 따라 자동 선택)
    resampler_config: Optional[ResamplerConfig] = Field(default=None, description="Resampler configuration")

    # Legacy 호환을 위한 개별 필드들 (deprecated, resampler_config 우선)
    latent_dimension: int = Field(default=768, description="Latent dimension")
    resampler_depth: Optional[int] = Field(default=None, description="Resampler depth")
    resampler_hidden_dim: Optional[int] = Field(default=None, description="Resampler hidden dimension")
    resampler_use_ln: bool = Field(default=True, description="Use LayerNorm in resampler")
    resampler_enable_cross_view: bool = Field(default=False, description="Enable cross-view attention")
    resampler_num_views: int = Field(default=8, description="Number of views")
    resampler_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")
    resampler_heads: Optional[int] = Field(default=None, description="Number of attention heads (Perceiver)")
    resampler_num_latents: Optional[int] = Field(default=None, description="Number of latent tokens (Perceiver)")
    resampler_num_query_tokens: Optional[int] = Field(default=None, description="Number of query tokens (QFormer resampler)")
    resampler_num_hidden_layers: Optional[int] = Field(default=None, description="Number of hidden layers (QFormer resampler)")
    resampler_attention_heads: Optional[int] = Field(default=None, description="Attention heads (QFormer resampler)")

    # VICReg 설정
    vicreg_loss_weight: float = Field(default=1.0, ge=0.0, description="VICReg loss weight")
    vicreg_similarity_weight: float = Field(default=25.0, description="VICReg similarity weight")
    vicreg_variance_weight: float = Field(default=25.0, description="VICReg variance weight")
    vicreg_covariance_weight: float = Field(default=1.0, description="VICReg covariance weight")
    vicreg_mode: str = Field(default="pairwise", description="VICReg calculation mode: 'pairwise' (default) or 'batchwise'")
    overlap_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Overlap ratio for panorama views")
    use_vicreg_norm: bool = Field(default=True, description="Use LayerNorm in VICReg path")

    # 이미지 처리
    image_size: Optional[tuple[int, int]] = Field(default=None, description="Image size (H, W)")
    max_text_length: int = Field(default=512, gt=0, description="Maximum text length")

    # 스티칭 설정
    stitching_mode: Literal["drop_overlap", "stride_views", "concat", "resample"] = Field(
        default="drop_overlap",
        description="Token stitching mode"
    )
    stitch_stride_offset: int = Field(default=0, ge=0, description="Stride offset for stitching")
    stitch_target_cols: int = Field(default=0, ge=0, description="Target columns (0=auto)")
    stitch_target_to_view_width: bool = Field(default=False, description="Match width to view width")
    stitch_interp: Literal["nearest", "linear"] = Field(default="nearest", description="Interpolation mode")

    # LoRA 설정
    use_lora: bool = Field(default=False, description="Enable LoRA")
    lora_r: int = Field(default=16, gt=0, description="LoRA rank")
    lora_alpha: int = Field(default=32, gt=0, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")
    lora_target_modules: Optional[List[str]] = Field(default=None, description="LoRA target modules")

    @model_validator(mode='before')
    @classmethod
    def apply_resampler_defaults(cls, values):
        """리샘플러 타입에 따른 기본값 자동 적용"""
        resampler_type = values.get('resampler_type', 'mlp')

        # resampler_config가 dict로 들어온 경우 타입에 맞게 변환
        if isinstance(values.get('resampler_config'), dict):
            rc_dict = values['resampler_config']
            if resampler_type == 'mlp':
                values['resampler_config'] = MLPResamplerConfig(**rc_dict)
            elif resampler_type == 'perceiver':
                values['resampler_config'] = PerceiverResamplerConfig(**rc_dict)
            elif resampler_type in ['bimamba', 'bidirectional_mamba', 'bi_mamba']:
                values['resampler_config'] = BiMambaResamplerConfig(**rc_dict)
            elif resampler_type == 'qformer':
                values['resampler_config'] = QFormerResamplerConfig(**rc_dict)

        # resampler_config가 없으면 타입에 맞게 자동 생성
        elif values.get('resampler_config') is None:
            resampler_defaults = RESAMPLER_CONFIGS.get(resampler_type, {})

            # 기본 파라미터 수집
            base_params = {
                'latent_dimension': values.get('latent_dimension', resampler_defaults.get('latent_dimension', 768)),
                'depth': values.get('resampler_depth', resampler_defaults.get('depth', 2)),
                'use_ln': values.get('resampler_use_ln', resampler_defaults.get('use_ln', True)),
                'dropout': values.get('resampler_dropout', resampler_defaults.get('dropout', 0.1)),
                'num_views': values.get('resampler_num_views', resampler_defaults.get('num_views', 8)),
            }

            # 타입별 config 생성
            if resampler_type == 'mlp':
                base_params['hidden_dim'] = values.get('resampler_hidden_dim', resampler_defaults.get('hidden_dim', 1536))
                values['resampler_config'] = MLPResamplerConfig(**base_params)

            elif resampler_type == 'perceiver':
                base_params['num_latents'] = values.get('resampler_num_latents', resampler_defaults.get('num_latents', 32))
                base_params['heads'] = values.get('resampler_heads', resampler_defaults.get('heads', 8))
                values['resampler_config'] = PerceiverResamplerConfig(**base_params)

            elif resampler_type in ['bimamba', 'bidirectional_mamba', 'bi_mamba']:
                base_params['hidden_dim'] = values.get('resampler_hidden_dim', resampler_defaults.get('hidden_dim', 1536))
                base_params['d_state'] = resampler_defaults.get('d_state', 64)
                base_params['d_conv'] = resampler_defaults.get('d_conv', 4)
                base_params['expand'] = resampler_defaults.get('expand', 2.0)
                base_params['norm_first'] = resampler_defaults.get('norm_first', True)
                base_params['enable_cross_view'] = values.get('resampler_enable_cross_view', resampler_defaults.get('enable_cross_view', False))
                values['resampler_config'] = BiMambaResamplerConfig(**base_params)
            elif resampler_type == 'qformer':
                base_params['hidden_dim'] = values.get('resampler_hidden_dim', resampler_defaults.get('hidden_dim', 768))
                base_params['num_query_tokens'] = values.get('resampler_num_query_tokens', resampler_defaults.get('num_query_tokens', 64))
                base_params['num_hidden_layers'] = values.get('resampler_num_hidden_layers', resampler_defaults.get('num_hidden_layers', 6))
                base_params['num_attention_heads'] = values.get('resampler_attention_heads', resampler_defaults.get('num_attention_heads', 8))
                values['resampler_config'] = QFormerResamplerConfig(**base_params)

        # Legacy 호환: resampler_config에서 개별 필드 동기화
        if values.get('resampler_config') is not None:
            rc = values['resampler_config']
            values['latent_dimension'] = rc.latent_dimension
            values['resampler_depth'] = rc.depth
            values['resampler_use_ln'] = rc.use_ln
            values['resampler_dropout'] = rc.dropout
            values['resampler_num_views'] = rc.num_views

            if isinstance(rc, MLPResamplerConfig):
                values['resampler_hidden_dim'] = rc.hidden_dim
            elif isinstance(rc, PerceiverResamplerConfig):
                values['resampler_num_latents'] = rc.num_latents
                values['resampler_heads'] = rc.heads
            elif isinstance(rc, BiMambaResamplerConfig):
                values['resampler_hidden_dim'] = rc.hidden_dim
                values['resampler_enable_cross_view'] = rc.enable_cross_view
            elif isinstance(rc, QFormerResamplerConfig):
                values['resampler_hidden_dim'] = rc.hidden_dim
                values['resampler_num_query_tokens'] = rc.num_query_tokens
                values['resampler_num_hidden_layers'] = rc.num_hidden_layers
                values['resampler_attention_heads'] = rc.num_attention_heads

        return values

    @field_validator('lora_target_modules', mode='before')
    @classmethod
    def set_lora_defaults(cls, v, info):
        """LoRA 타겟 모듈 기본값 설정"""
        if v is None and info.data.get('use_lora', False):
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        return v

    def get_model_kwargs(self) -> Dict[str, Any]:
        """PanoramaVLM 모델 생성에 필요한 kwargs 반환"""
        return {
            'vision_name': self.vision_name,
            'language_model_name': self.language_model_name,
            'resampler_type': self.resampler_type,
            'latent_dimension': self.latent_dimension,
            'vicreg_loss_weight': self.vicreg_loss_weight,
            'overlap_ratio': self.overlap_ratio,
            'use_vicreg_norm': self.use_vicreg_norm,
            'max_text_length': self.max_text_length,
            'vicreg_similarity_weight': self.vicreg_similarity_weight,
            'vicreg_variance_weight': self.vicreg_variance_weight,
            'vicreg_covariance_weight': self.vicreg_covariance_weight,
            'stitching_mode': self.stitching_mode,
            'stitch_stride_offset': self.stitch_stride_offset,
            'stitch_target_cols': self.stitch_target_cols,
            'stitch_target_to_view_width': self.stitch_target_to_view_width,
            'stitch_interp': self.stitch_interp,
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (legacy 호환)"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 생성 (legacy 호환)"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """설정 유효성 검사 (legacy 호환)"""
        try:
            self.model_validate(self.model_dump())
            return True
        except Exception as e:
            warnings.warn(f"Validation failed: {e}")
            return False

    class Config:
        extra = "allow"  # 하위 호환성을 위해 추가 필드 허용
        validate_assignment = False  # 무한 재귀 방지


class ImageProcessingConfig(BaseModel):
    """이미지 처리 설정"""
    crop_strategy: Literal["e2p", "cubemap", "none"] = Field(default="e2p")
    image_size: List[int] = Field(default=[224, 224])
    overlap_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    stitching_mode: str = Field(default="concat")
    stitch_target_to_view_width: bool = Field(default=True)
    stitch_interp: str = Field(default="linear")
    fov_deg: float = Field(default=90.0, gt=0.0)
    use_vision_processor: bool = Field(default=True)
    anyres_patch_size: int = Field(default=336, gt=0)
    anyres_max_patches: int = Field(default=12, gt=0)
    normalize: bool = Field(default=True)
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None

    class Config:
        extra = "allow"


class StageDataConfig(BaseModel):
    """스테이지별 데이터 설정"""
    csv_train: Union[str, List[str]]
    csv_val: Union[str, List[str]]

    class Config:
        extra = "allow"


class StageConfig(BaseModel):
    """스테이지별 훈련 설정"""
    epochs: int = Field(default=1, gt=0)
    lr: float = Field(default=2e-5, gt=0.0)
    batch_size: int = Field(default=1, gt=0)
    accumulate_grad_batches: int = Field(default=2, gt=0)
    vicreg_loss_weight: float = Field(default=0.0, ge=0.0)
    max_text_length: Union[int, str] = Field(default=256)
    data: Optional[StageDataConfig] = None

    @classmethod
    def for_vision(cls) -> 'StageConfig':
        """Vision 단계 기본 설정"""
        return cls(
            epochs=5,
            lr=1e-5,
            batch_size=64,
            accumulate_grad_batches=2,
            vicreg_loss_weight=1.0,
            max_text_length=32,
        )

    @classmethod
    def for_resampler(cls) -> 'StageConfig':
        """Resampler 단계 기본 설정"""
        return cls(
            epochs=1,
            lr=2e-6,
            batch_size=1,
            accumulate_grad_batches=2,
            vicreg_loss_weight=0.0,
            max_text_length=256,
        )

    @classmethod
    def for_finetune(cls) -> 'StageConfig':
        """Finetune 단계 기본 설정"""
        return cls(
            epochs=1,
            lr=2e-6,
            batch_size=1,
            accumulate_grad_batches=2,
            vicreg_loss_weight=0.0,
            max_text_length=256,
        )

    @classmethod
    def get_default(cls, stage: str) -> 'StageConfig':
        """스테이지별 기본 설정 가져오기"""
        factory_map = {
            'vision': cls.for_vision,
            'resampler': cls.for_resampler,
            'finetune': cls.for_finetune,
        }
        factory = factory_map.get(stage)
        if factory:
            return factory()
        return cls()  # 기본 StageConfig 반환

    class Config:
        extra = "allow"


class TrainingConfig(BaseModel):
    """훈련 전역 설정"""
    prefix: str = Field(default="panovlm")
    stages: List[str] = Field(default=["vision", "resampler", "finetune"])
    num_workers: int = Field(default=16, ge=0)
    system_msg: str = Field(default="You are a helpful assistant. Describe the panorama image.")
    wandb_project: str = Field(default="panollava-training")
    empty_cache_each_step: int = Field(default=1, ge=0)
    stage_configs: Dict[str, Union[StageConfig, Dict[str, Any]]] = Field(default_factory=dict)

    # DeepSpeed
    deepspeed: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "strategy": {"stage": 3}}
    )

    class Config:
        extra = "allow"


class ExperimentConfig(BaseModel):
    """실험 메타 정보"""
    name: str = Field(default="experiment")
    description: str = Field(default="")
    version: str = Field(default="1.0")

    class Config:
        extra = "allow"


class PathsConfig(BaseModel):
    """경로 설정"""
    runs_dir: str = Field(default="runs")
    csv_train: Optional[Union[str, List[str]]] = None
    csv_val: Optional[Union[str, List[str]]] = None

    class Config:
        extra = "allow"


class EnvironmentConfig(BaseModel):
    """환경 설정"""
    cuda_visible_devices: str = Field(default="0")
    wandb_project: str = Field(default="panollava-training")
    output_dir: Optional[str] = None

    class Config:
        extra = "allow"


class LoRAConfig(BaseModel):
    """LoRA 설정"""
    use_lora: bool = Field(default=False)
    rank: int = Field(default=32, gt=0)
    alpha: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    save_lora_only: bool = Field(default=False)

    class Config:
        extra = "allow"


class PanoVLMConfig(BaseModel):
    """통합 설정 루트"""
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: Optional[Dict[str, Any]] = None
    system_messages: Optional[Dict[str, str]] = None
    generation: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'PanoVLMConfig':
        """YAML 파일에서 설정 로드"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_legacy_json(cls, json_path: Union[str, Path]) -> 'PanoVLMConfig':
        """Legacy JSON에서 설정 로드 (하위 호환성)"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        warnings.warn(
            "Loading from legacy JSON format. Please migrate to YAML.",
            DeprecationWarning
        )

        return cls(**config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """YAML 파일로 저장"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump(exclude_none=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)

        print(f"✅ Config saved: {yaml_path}")

    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """특정 스테이지의 설정 가져오기 (기본값 병합)"""
        # 스테이지 기본값 (StageConfig 클래스에서 가져오기)
        default_config = StageConfig.get_default(stage)
        defaults = default_config.model_dump(exclude_none=True)

        # YAML 설정
        stage_config = self.training.stage_configs.get(stage, {})

        # 병합 (YAML 우선)
        if isinstance(stage_config, StageConfig):
            merged = {**defaults, **stage_config.model_dump(exclude_none=True)}
        else:
            merged = {**defaults, **stage_config}

        return merged

    class Config:
        extra = "allow"  # 하위 호환성
        validate_assignment = True


# ============================================================================
# Legacy 호환 클래스 (기존 코드 지원)
# ============================================================================

class Config:
    """Legacy Config 클래스 (하위 호환성 - train.py용)"""
    RESAMPLER_DEFAULTS = RESAMPLER_CONFIGS
    STAGE_DEFAULTS = None  # Will be initialized after model_rebuild()

    @staticmethod
    def _get_stage_defaults():
        """Lazy evaluation for STAGE_DEFAULTS"""
        return {
            "vision": StageConfig.for_vision().model_dump(exclude_none=True),
            "resampler": StageConfig.for_resampler().model_dump(exclude_none=True),
            "finetune": StageConfig.for_finetune().model_dump(exclude_none=True),
        }


class ConfigManager:
    """Legacy ConfigManager (하위 호환성 - train.py, model.py용)"""

    @staticmethod
    def load_config(file_path: Union[str, Path]) -> ModelConfig:
        """Legacy: JSON/YAML에서 ModelConfig 로드"""
        file_path = Path(file_path)

        if file_path.suffix.lower() in {'.yaml', '.yml'}:
            full_config = PanoVLMConfig.from_yaml(file_path)
            return full_config.models
        else:
            full_config = PanoVLMConfig.from_legacy_json(file_path)
            return full_config.models

    @staticmethod
    def save_config(config: ModelConfig, file_path: Union[str, Path]) -> None:
        """Legacy: ModelConfig를 JSON으로 저장"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(exclude_none=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ Config saved: {file_path}")

    @staticmethod
    def _flatten_json_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy: nested config를 flat하게 변환 (train.py용)"""
        flat = {}

        # models 블록
        if "models" in config_dict:
            flat.update(config_dict["models"])

        # image_processing 블록
        if "image_processing" in config_dict:
            img_proc = config_dict["image_processing"]
            flat["overlap_ratio"] = img_proc.get("overlap_ratio", 0.5)
            flat["stitching_mode"] = img_proc.get("stitching_mode", "concat")
            flat["stitch_target_to_view_width"] = img_proc.get("stitch_target_to_view_width", False)
            flat["stitch_interp"] = img_proc.get("stitch_interp", "linear")

        # lora 블록
        if "lora" in config_dict:
            lora = config_dict["lora"]
            flat["use_lora"] = lora.get("use_lora", False)
            flat["lora_r"] = lora.get("rank", 16)
            flat["lora_alpha"] = lora.get("alpha", 32)
            flat["lora_dropout"] = lora.get("dropout", 0.1)
            flat["lora_target_modules"] = lora.get("target_modules")

        # training 블록
        if "training" in config_dict:
            training = config_dict["training"]
            flat["max_text_length"] = training.get("max_text_length", 512)

        return flat


# ============================================================================
# Model rebuild for forward references (Python 3.13 + Pydantic v2)
# ============================================================================

ModelConfig.model_rebuild()
PanoVLMConfig.model_rebuild()
StageConfig.model_rebuild()

# Initialize legacy Config.STAGE_DEFAULTS after model_rebuild()
Config.STAGE_DEFAULTS = Config._get_stage_defaults()
