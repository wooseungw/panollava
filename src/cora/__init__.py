"""CORA: Panoramic Vision-Language Model Framework.

Unified package for panoramic VLM research, combining custom CORA architecture
and baseline commercial VLM LoRA finetuning.

Subpackages::

    cora.model       – PanoramaVLM, VisionBackbone, resamplers, projectors
    cora.training    – CORATrainer, PanoramaTrainingModule, VICRegLoss
    cora.data        – PanoramaDataset, PanoramaDataModule
    cora.evaluation  – CORAEvaluator (5 metrics + CSV)
    cora.processors  – PanoramaProcessor, PanoramaImageProcessor
    cora.inference   – PanoramaGenerator
    cora.baseline    – BaselineTrainer, BaselineModelRegistry
    cora.config      – CORAConfig, ConfigManager
"""

__version__ = "2.0.0"
