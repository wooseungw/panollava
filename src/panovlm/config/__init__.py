from __future__ import annotations

"""
개선된 설정 관리 시스템
"""

"""Unified config namespace exposing both legacy and new-style helpers."""

import importlib.util
from pathlib import Path
from typing import Any

from .config_manager import ConfigManager as YAMLConfigManager, ExperimentConfig
from .stage_config import StageConfig, TrainingStageConfig, LossConfig, DatasetConfig

globals()['YAMLConfigManager'] = YAMLConfigManager

__all__ = [
    'YAMLConfigManager',
    'ExperimentConfig',
    'StageConfig',
    'TrainingStageConfig',
    'LossConfig',
    'DatasetConfig',
]


def _load_legacy_model_config() -> dict[str, Any]:
    """Load legacy `ModelConfig` definitions from the pre-package module.

    Legacy tooling (train/eval scripts, notebooks) expect ``panovlm.config`` to
    expose :class:`ModelConfig` and helper utilities that historically lived in
    ``src/panovlm/config.py``.  Once the directory-based config package was
    added, that flat module became shadowed and those imports broke.  To stay
    backward compatible, we lazily load the legacy module via an importlib
    shim and re-export the expected symbols.
    """

    legacy_path = Path(__file__).resolve().parent.parent / 'config.py'
    if not legacy_path.is_file():
        return {}

    spec = importlib.util.spec_from_file_location(
        'panovlm._legacy_model_config', legacy_path
    )
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exported = {}
    for name in ('ModelConfig', 'ConfigManager', 'Config'):
        if hasattr(module, name):
            exported[name] = getattr(module, name)
    return exported


_legacy_symbols = _load_legacy_model_config()
for _name, _value in _legacy_symbols.items():
    if _name in {"ModelConfig", "ConfigManager", "Config"}:
        globals()[_name] = _value
    else:
        globals().setdefault(_name, _value)
__all__.extend(name for name in _legacy_symbols if name not in __all__)
