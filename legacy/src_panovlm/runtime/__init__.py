"""Runtime helpers for PanoramaVLM.

This package centralizes configuration loading, stage orchestration, and
model construction utilities so training/evaluation scripts can stay thin.
"""

from panovlm.config.loader import RuntimeConfigBundle, load_config_dict, load_runtime_config  # noqa: F401
from .stage_manager import (
    StageManager,
    StageDefinition,
    STAGE_ALIAS_MAP,
    canonical_stage_name,
)  # noqa: F401
from .model_factory import ModelFactory  # noqa: F401
