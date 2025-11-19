"""Stage orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from panovlm.config import Config, StageConfig


STAGE_ALIAS_MAP = {
    "vision": "vision",
    "vision_pretraining": "vision",
    "vision_pretrain": "vision",
    "joint_vision_resampler": "vision",
    "joint_resampler": "vision",
    "resampler": "resampler",
    "resampler_training": "resampler",
    "finetune": "finetune",
    "instruction_tuning": "finetune",
    "instruction_tune": "finetune",
    "final_finetune": "finetune",
    "generate": "generate",
    "inference": "generate",
}


def canonical_stage_name(stage: str) -> str:
    key = str(stage).strip()
    canonical = STAGE_ALIAS_MAP.get(key)
    if canonical is None:
        valid = ", ".join(sorted(STAGE_ALIAS_MAP.keys()))
        raise ValueError(f"stage는 [{valid}] 중 하나여야 합니다 (got: {stage})")
    return canonical


@dataclass
class StageDefinition:
    name: str
    canonical_name: str
    config: Dict[str, Any]


class StageManager:
    """Encapsulates stage list resolution and per-stage config merges."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ── Stage enumeration -------------------------------------------------
    def available_stage_names(self) -> List[str]:
        override = self.cfg.get("_cli_stage_override")
        if isinstance(override, list) and override:
            return override
        training_cfg = self.cfg.get("training", {}) or {}
        stages = training_cfg.get("stages")
        if isinstance(stages, list) and stages:
            return stages
        default_stage = training_cfg.get("default_stage")
        if isinstance(default_stage, str) and default_stage:
            return [default_stage]
        return ["vision"]

    def resolve_stage_override(self, stage_arg: Optional[str]) -> Optional[List[str]]:
        """Parse CLI overrides like 'vision,resampler' or '1,3'."""
        if stage_arg is None:
            return None
        stage_arg = stage_arg.strip()
        if not stage_arg or stage_arg.lower() == "all":
            return None

        tokens = [token.strip() for token in stage_arg.split(",") if token.strip()]
        if not tokens:
            return None

        available = self.available_stage_names()
        lower_map = {name.lower(): name for name in available}
        resolved: List[str] = []

        for token in tokens:
            if token.isdigit():
                idx = int(token) - 1
                if idx < 0 or idx >= len(available):
                    raise ValueError(f"Stage index {token} is out of range (1-{len(available)})")
                resolved.append(available[idx])
                continue

            key = token.lower()
            match = lower_map.get(key)
            if match is None:
                alias_target = STAGE_ALIAS_MAP.get(key)
                if alias_target:
                    for candidate in available:
                        if canonical_stage_name(candidate) == alias_target:
                            match = candidate
                            break
            if match is None:
                raise ValueError(f"Unknown stage '{token}'. Available stages: {available}")
            resolved.append(match)

        return resolved

    # ── Per-stage config --------------------------------------------------
    def stage_configs(self) -> Dict[str, Any]:
        training_cfg = self.cfg.get("training", {}) or {}
        return training_cfg.get("stage_configs") or {}

    def _stage_block(self, stage: str) -> Dict[str, Any]:
        stage_configs = self.stage_configs()
        canonical = STAGE_ALIAS_MAP.get(stage, stage)
        yaml_config = stage_configs.get(stage) or stage_configs.get(canonical)
        if not yaml_config:
            available = sorted(stage_configs.keys())
            raise ValueError(f"Stage '{stage}' 설정을 YAML에서 찾을 수 없습니다. 정의된 단계: {available}")
        if isinstance(yaml_config, StageConfig):
            return yaml_config.model_dump(exclude_none=True)
        if isinstance(yaml_config, dict):
            return dict(yaml_config)
        raise TypeError(f"Stage '{stage}' 설정 형식이 잘못되었습니다: {type(yaml_config)}")

    def get_stage_definition(self, stage: str) -> StageDefinition:
        config = self._stage_block(stage)
        canonical = STAGE_ALIAS_MAP.get(stage, stage)
        self._validate_stage_config(stage, config)
        return StageDefinition(name=stage, canonical_name=canonical, config=config)

    def _validate_stage_config(self, stage: str, stage_cfg: Dict[str, Any]) -> None:
        required_stage_keys = ("epochs", "lr", "batch_size")
        missing_stage_keys = [key for key in required_stage_keys if stage_cfg.get(key) is None]
        if missing_stage_keys:
            raise ValueError(f"Stage '{stage}' 설정에 필수 키가 누락되었습니다: {missing_stage_keys}")
        if "lr" in stage_cfg and isinstance(stage_cfg["lr"], str):
            try:
                stage_cfg["lr"] = float(stage_cfg["lr"])
            except ValueError:
                raise ValueError(f"Stage '{stage}' has invalid lr value: {stage_cfg['lr']}") from None

    def iter_stage_definitions(self, stage_names: Optional[Iterable[str]] = None) -> List[StageDefinition]:
        names = list(stage_names) if stage_names is not None else self.available_stage_names()
        return [self.get_stage_definition(name) for name in names]

    # ── Diagnostics -------------------------------------------------------
    def preview(self) -> List[Dict[str, Any]]:
        preview_data: List[Dict[str, Any]] = []
        for definition in self.iter_stage_definitions():
            cfg = definition.config
            summary = {
                "stage": definition.name,
                "canonical": definition.canonical_name,
                "epochs": cfg.get("epochs"),
                "lr": cfg.get("lr"),
                "batch_size": cfg.get("batch_size"),
                "accumulate_grad_batches": cfg.get("accumulate_grad_batches"),
                "data": cfg.get("data"),
            }
            preview_data.append(summary)
        return preview_data
