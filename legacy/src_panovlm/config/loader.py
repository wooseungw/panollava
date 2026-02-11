"""Unified configuration loader for training/evaluation scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .schema import PanoVLMConfig, ModelConfig


def _normalize_dataset_dict(dataset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize legacy dataset blocks into the csv_train/csv_val shape."""
    if not isinstance(dataset, dict):
        return {}
    normalized = dict(dataset)
    if "train_csv" in normalized and "csv_train" not in normalized:
        normalized["csv_train"] = normalized["train_csv"]
    if "val_csv" in normalized and "csv_val" not in normalized:
        normalized["csv_val"] = normalized["val_csv"]
    if "train" not in normalized and isinstance(normalized.get("csv_train"), list):
        normalized["train"] = normalized.get("csv_train")
    if "val" not in normalized and isinstance(normalized.get("csv_val"), list):
        normalized["val"] = normalized.get("csv_val")
    return normalized


def _convert_yaml_config(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal normalization so the rest of the code can treat the YAML as flat dicts."""
    result: Dict[str, Any] = {}

    # Copy the top-level blocks we care about verbatim.
    for key in (
        "experiment",
        "models",
        "image_processing",
        "environment",
        "data",
        "paths",
        "system_messages",
        "lora",
        "generation",
    ):
        if key in yaml_cfg:
            result[key] = yaml_cfg[key]

    # training block normalization
    training_block = yaml_cfg.get("training", {}) or {}

    if isinstance(training_block.get("stages"), list):
        result["training"] = dict(training_block)
    else:
        result["training"] = dict(training_block)
        result["training"].setdefault("stages", ["vision", "resampler", "finetune"])

    stage_configs = training_block.get("stage_configs", {})
    if isinstance(stage_configs, dict):
        result["training"]["stage_configs"] = stage_configs

    # paths defaults
    if "paths" not in result:
        result["paths"] = {}
    result["paths"].setdefault("runs_dir", yaml_cfg.get("environment", {}).get("output_dir", "runs"))

    # pull csv references from legacy data section if present
    result["paths"].setdefault("csv_train", yaml_cfg.get("paths", {}).get("csv_train"))
    result["paths"].setdefault("csv_val", yaml_cfg.get("paths", {}).get("csv_val"))

    data_block = yaml_cfg.get("data", {}) or {}
    if data_block:
        if "train" in data_block:
            result["paths"].setdefault("csv_train", data_block["train"])
        if "val" in data_block:
            result["paths"].setdefault("csv_val", data_block["val"])

    result.setdefault("data", _normalize_dataset_dict(data_block))
    return result


def load_config_dict(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the training YAML into a normalized dictionary."""
    env_path = os.environ.get("PANOVLM_CONFIG")
    cfg_path = config_path or env_path or "config.yaml"
    path = Path(cfg_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    os.environ["PANOVLM_CONFIG"] = str(path)

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Only YAML configuration files are supported. Please provide a .yaml/.yml file.")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configs, but it is not installed.")

    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if not isinstance(raw_cfg, dict):
        raise ValueError("YAML config must deserialize to a mapping")

    cfg = _convert_yaml_config(raw_cfg)
    cfg["_config_path"] = str(path)
    return cfg


@dataclass
class RuntimeConfigBundle:
    """Container with both the normalized dict and typed config objects."""

    raw: Dict[str, Any]
    pano: PanoVLMConfig
    model: ModelConfig

    @property
    def stage_configs(self) -> Dict[str, Any]:
        return self.raw.get("training", {}).get("stage_configs", {})


def load_runtime_config(config_path: Optional[str] = None) -> RuntimeConfigBundle:
    """Convenience helper for scripts that need both dict + typed configs."""
    raw_cfg = load_config_dict(config_path)
    pano_cfg = PanoVLMConfig(**raw_cfg)
    model_cfg = pano_cfg.models
    # cache the serialized ModelConfig so repeated invocations are cheap
    raw_cfg["_model_config_obj"] = model_cfg
    return RuntimeConfigBundle(raw=raw_cfg, pano=pano_cfg, model=model_cfg)
