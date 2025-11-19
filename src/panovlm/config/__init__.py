"""
Centralized configuration namespace for PanoramaVLM.

This package now hosts every config-related helper (schema definitions,
manager utilities, and legacy compatibility shims) so downstream code can
simply import from ``panovlm.config`` without caring about the underlying
layout.
"""

from .schema import (  # noqa: F401
    RESAMPLER_CONFIGS,
    BiMambaResamplerConfig,
    Config,
    ConfigManager,
    ConfigManager as LegacyConfigManager,
    ImageProcessingConfig,
    LoRAConfig,
    MLPResamplerConfig,
    ModelConfig,
    PanoVLMConfig,
    PathsConfig,
    PerceiverResamplerConfig,
    QFormerResamplerConfig,
    ResamplerConfig,
    ResamplerConfigBase,
    StageConfig,
    StageDataConfig,
    TrainingConfig,
)

from .manager import ExperimentConfig, UnifiedConfigManager

YAMLConfigManager = UnifiedConfigManager

__all__ = [
    "RESAMPLER_CONFIGS",
    "BiMambaResamplerConfig",
    "Config",
    "ConfigManager",
    "LegacyConfigManager",
    "ImageProcessingConfig",
    "LoRAConfig",
    "MLPResamplerConfig",
    "ModelConfig",
    "PanoVLMConfig",
    "PathsConfig",
    "PerceiverResamplerConfig",
    "QFormerResamplerConfig",
    "ResamplerConfig",
    "ResamplerConfigBase",
    "StageConfig",
    "StageDataConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "UnifiedConfigManager",
    "YAMLConfigManager",
]
