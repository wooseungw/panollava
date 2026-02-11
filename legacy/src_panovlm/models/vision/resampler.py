"""Resampler wrappers for panoramic features."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from copy import deepcopy

from ..resampler.resamplers import MLPResampler, QFormerResampler
import logging

try:
    from ..resampler.bimamba import BidirectionalMambaResampler  # type: ignore
    _BIMAMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BidirectionalMambaResampler = None  # type: ignore
    _BIMAMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


CANONICAL_RESAMPLER_TYPES = {
    "mlp": "mlp",
    "bimamba": "bimamba",
    "bidirectional_mamba": "bimamba",
    "bi_mamba": "bimamba",
    "qformer": "qformer",
}

# YAML에서는 타입만 선택하고, 각 리샘플러의 세부 파라미터는 아래 사전을 직접 수정해 관리합니다.
RESAMPLER_PRESETS = {
    "mlp": {
        "hidden_dim": None,
        "depth": 3,
        "use_ln": True,
        "pool_tokens": None,
        "pool_type": "avg",
    },
    "bimamba": {
        "hidden_dim": None,
        "num_layers": 4,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2.0,
        "dropout": 0.0,
        "norm_first": True,
    },
    "qformer": {
        "hidden_dim": None,
        "num_query_tokens": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
    },
}


class ResamplerModule(nn.Module):
    def __init__(self, config: Any, input_dim: int):
        super().__init__()
        resampler_type = getattr(config, 'resampler_type', 'mlp')
        canonical_type = CANONICAL_RESAMPLER_TYPES.get(resampler_type, resampler_type)
        latent_dimension = int(getattr(config, 'latent_dimension'))  # 필수 설정

        preset = RESAMPLER_PRESETS.get(canonical_type)
        if preset is None:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {resampler_type}")
        preset_kwargs = deepcopy(preset)

        def _set_from_attr(attr_name: str, target_key: str) -> None:
            value = getattr(config, attr_name, None)
            if value is not None:
                preset_kwargs[target_key] = value

        # 1) 상위 ModelConfig 필드로부터 공통 override 적용 (resampler_config 이전)
        common_attr_map = (
            ("resampler_depth", "depth"),
            ("resampler_hidden_dim", "hidden_dim"),
            ("resampler_use_ln", "use_ln"),
            ("resampler_pool_tokens", "pool_tokens"),
            ("resampler_pool_type", "pool_type"),
            ("resampler_dropout", "dropout"),
            ("resampler_num_views", "num_views"),
        )
        for attr_name, target in common_attr_map:
            _set_from_attr(attr_name, target)

        if canonical_type == "perceiver":
            _set_from_attr("resampler_num_latents", "num_latents")
            _set_from_attr("resampler_heads", "heads")
        elif canonical_type in {"bimamba", "bidirectional_mamba", "bi_mamba"}:
            _set_from_attr("resampler_enable_cross_view", "enable_cross_view")
        elif canonical_type == "qformer":
            _set_from_attr("resampler_num_query_tokens", "num_query_tokens")
            _set_from_attr("resampler_num_hidden_layers", "num_hidden_layers")
            _set_from_attr("resampler_attention_heads", "num_attention_heads")

        resampler_cfg = getattr(config, 'resampler_config', None)
        cfg_dict: dict[str, object] = {}
        if resampler_cfg is not None:
            if hasattr(resampler_cfg, 'model_dump'):
                cfg_dict = resampler_cfg.model_dump(exclude_none=True)
            elif isinstance(resampler_cfg, dict):
                cfg_dict = {k: v for k, v in resampler_cfg.items() if v is not None}

        if cfg_dict:
            for key in ('depth', 'hidden_dim', 'use_ln', 'dropout', 'num_views', 'pool_tokens', 'pool_type'):
                if key in cfg_dict:
                    preset_kwargs[key] = cfg_dict[key]

            if canonical_type in {"bimamba", "bidirectional_mamba", "bi_mamba"}:
                for key in ('d_state', 'd_conv', 'expand', 'norm_first'):
                    if key in cfg_dict:
                        preset_kwargs[key] = cfg_dict[key]

        # alias support: depth may be provided as num_layers
        if 'num_layers' in cfg_dict and 'depth' not in preset_kwargs:
            preset_kwargs['depth'] = cfg_dict['num_layers']

        if canonical_type == "mlp":
            self.resampler = MLPResampler(
                input_dim,
                latent_dimension,
                hidden_dim=preset_kwargs.get("hidden_dim"),
                depth=int(preset_kwargs.get("depth", 3)),
                use_ln=preset_kwargs.get("use_ln", True),
                pool_tokens=preset_kwargs.get("pool_tokens"),
                pool_type=str(preset_kwargs.get("pool_type", "avg")),
            )
        elif canonical_type == "bimamba":
            if not _BIMAMBA_AVAILABLE or BidirectionalMambaResampler is None:
                logger.warning(
                    "BidirectionalMambaResampler requires the mamba-ssm package. "
                    "MLP 리샘플러로 자동 대체합니다. (`pip install mamba-ssm`)"
                )
                self.resampler = MLPResampler(
                    input_dim,
                    latent_dimension,
                    hidden_dim=preset_kwargs.get("hidden_dim"),
                    depth=RESAMPLER_PRESETS["mlp"].get("depth", 3),
                    use_ln=RESAMPLER_PRESETS["mlp"].get("use_ln", True),
                )
            else:
                hidden_dim = preset_kwargs.get("hidden_dim") or latent_dimension
                num_layers = int(preset_kwargs.get("depth", preset_kwargs.get("num_layers", 4)))
                self.resampler = BidirectionalMambaResampler(
                    input_dim,
                    latent_dimension,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    d_state=int(preset_kwargs.get("d_state", 64)),
                    d_conv=int(preset_kwargs.get("d_conv", 4)),
                    expand=float(preset_kwargs.get("expand", 2.0)),
                    dropout=float(preset_kwargs.get("dropout", 0.0)),
                    norm_first=bool(preset_kwargs.get("norm_first", True)),
                )
        elif canonical_type == "qformer":
            num_query = int(preset_kwargs.get("num_query_tokens", 64))
            num_layers = int(preset_kwargs.get("num_hidden_layers", preset_kwargs.get("depth", 6)))
            self.resampler = QFormerResampler(
                input_dim,
                latent_dimension,
                num_query=num_query,
                num_hidden_layers=num_layers,
            )
        self.output_dim = latent_dimension

    def forward(self, vision_features: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if target_dtype is not None and vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        return self.resampler(vision_features)
