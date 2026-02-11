"""High-level helpers for loading and manipulating PanoramaVLM configs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .loader import RuntimeConfigBundle, load_runtime_config
from .schema import ModelConfig, PanoVLMConfig, StageConfig


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into a copy of base."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[assignment]
        else:
            merged[key] = value
    return merged


@dataclass
class ExperimentConfig:
    """Wrapper providing convenient dict-style access to a validated config."""

    pano: PanoVLMConfig
    raw: Dict[str, Any] = field(default_factory=dict)
    stage_name: Optional[str] = None
    stage_config: Optional[StageConfig] = None

    @property
    def global_config(self) -> Dict[str, Any]:
        return self.raw.get("training", {}) or {}

    @property
    def model_config(self) -> Dict[str, Any]:
        return self.pano.models.model_dump(exclude_none=True)

    @property
    def data_config(self) -> Dict[str, Any]:
        return self.raw.get("data", {}) or {}

    @property
    def generation_config(self) -> Dict[str, Any]:
        return self.raw.get("generation", {}) or {}

    @property
    def environment_config(self) -> Dict[str, Any]:
        return self.raw.get("environment", {}) or {}

    @property
    def lora_config(self) -> Dict[str, Any]:
        return self.raw.get("lora", {}) or {}


class UnifiedConfigManager:
    """Single entry point for loading and inspecting PanoramaVLM configs."""

    def __init__(self, config_path: Union[str, Path], *, auto_load: bool = False):
        self.config_path = Path(config_path)
        self._bundle: Optional[RuntimeConfigBundle] = None
        if auto_load:
            self.load_config()

    @property
    def pano(self) -> PanoVLMConfig:
        if not self._bundle:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self._bundle.pano

    @property
    def model(self) -> ModelConfig:
        if not self._bundle:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self._bundle.model

    @property
    def raw(self) -> Dict[str, Any]:
        if not self._bundle:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self._bundle.raw

    def load_config(self) -> ExperimentConfig:
        """Load and validate the YAML file."""
        bundle = load_runtime_config(str(self.config_path))
        self._bundle = bundle
        return ExperimentConfig(pano=bundle.pano, raw=bundle.raw)

    def override_config(self, overrides: Dict[str, Any]) -> "UnifiedConfigManager":
        """Apply in-memory overrides and rebuild the typed config."""
        if not self._bundle:
            self.load_config()
        merged = _deep_merge(self._bundle.raw, overrides)
        pano_cfg = PanoVLMConfig(**merged)
        self._bundle = RuntimeConfigBundle(raw=merged, pano=pano_cfg, model=pano_cfg.models)
        return self

    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """Return a StageConfig model for the requested stage."""
        if not self._bundle:
            self.load_config()
        stage_configs = (self._bundle.raw.get("training", {}) or {}).get("stage_configs") or {}
        if stage_name not in stage_configs:
            return None
        stage_dict = self._bundle.pano.get_stage_config(stage_name)
        return StageConfig(**stage_dict)

    def get_available_stages(self) -> List[str]:
        if not self._bundle:
            self.load_config()
        training_block = self._bundle.raw.get("training", {}) or {}
        stage_configs = training_block.get("stage_configs") or {}
        if isinstance(stage_configs, dict) and stage_configs:
            return list(stage_configs.keys())
        stages = training_block.get("stages")
        if isinstance(stages, list):
            return stages
        default_stage = training_block.get("default_stage")
        if isinstance(default_stage, str):
            return [default_stage]
        return []

    def get_default_stage(self) -> Optional[str]:
        if not self._bundle:
            self.load_config()
        training_block = self._bundle.raw.get("training", {}) or {}
        default_stage = training_block.get("default_stage")
        if isinstance(default_stage, str):
            return default_stage
        stages = training_block.get("stages")
        if isinstance(stages, list) and stages:
            return stages[0]
        stage_cfgs = training_block.get("stage_configs")
        if isinstance(stage_cfgs, dict) and stage_cfgs:
            return next(iter(stage_cfgs.keys()))
        return None

    def get_experiment_config(self, stage_name: Optional[str] = None) -> ExperimentConfig:
        if not self._bundle:
            self.load_config()
        stage_cfg = self.get_stage_config(stage_name) if stage_name else None
        return ExperimentConfig(
            pano=self._bundle.pano,
            raw=self._bundle.raw,
            stage_name=stage_name,
            stage_config=stage_cfg,
        )

    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Serialize the (possibly overridden) config to YAML."""
        if not self._bundle:
            self.load_config()
        save_path = Path(output_path) if output_path else self.config_path
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._bundle.pano.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
                allow_unicode=True,
            )
        return save_path

    def as_dict(self) -> Dict[str, Any]:
        if not self._bundle:
            self.load_config()
        return deepcopy(self._bundle.raw)
