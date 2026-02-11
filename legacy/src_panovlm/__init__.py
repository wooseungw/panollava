"""PanoLLaVA - Panoramic Vision-Language Model

Main exports:
- PanoramaVLM: Main model class
- Config, ModelConfig, StageConfig: Configuration classes
- VLMDataModule, ChatPanoDataset: Dataset classes
- PanoramaImageProcessor, UniversalTextFormatter: Processors
"""

# Model
from .models.model import PanoramaVLM

# Configuration - import from config package (which loads from config.py)
from .config import ModelConfig, Config, ConfigManager, YAMLConfigManager, StageConfig

# Data
from .dataset import VLMDataModule, BaseChatPanoDataset, ChatPanoDataset, ChatPanoTestDataset

# Processors
from .processors import (
    PanoramaImageProcessor,
    VisionProcessorWrapper,
    PanoLLaVAProcessor,
    UniversalTextFormatter,
)

__all__ = [
    # Model
    "PanoramaVLM",
    # Config
    "Config",
    "ModelConfig",
    "StageConfig",
    "ConfigManager",
    "YAMLConfigManager",
    # Data
    "VLMDataModule",
    "BaseChatPanoDataset",
    "ChatPanoDataset",
    "ChatPanoTestDataset",
    # Processors
    "PanoramaImageProcessor",
    "VisionProcessorWrapper",
    "PanoLLaVAProcessor",
    "UniversalTextFormatter",
]
