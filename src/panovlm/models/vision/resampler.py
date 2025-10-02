"""Resampler wrappers for panoramic features."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from copy import deepcopy

from ..resampler.resamplers import MLPResampler
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
}

# YAML에서는 타입만 선택하고, 각 리샘플러의 세부 파라미터는 아래 사전을 직접 수정해 관리합니다.
RESAMPLER_PRESETS = {
    "mlp": {
        "hidden_dim": None,
        "depth": 3,
        "use_ln": True,
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

        if canonical_type == "mlp":
            self.resampler = MLPResampler(
                input_dim,
                latent_dimension,
                hidden_dim=preset_kwargs.get("hidden_dim"),
                depth=preset_kwargs.get("depth", 3),
                use_ln=preset_kwargs.get("use_ln", True),
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
                hidden_dim = preset_kwargs.pop("hidden_dim", None) or latent_dimension
                self.resampler = BidirectionalMambaResampler(
                    input_dim,
                    latent_dimension,
                    hidden_dim=hidden_dim,
                    num_layers=preset_kwargs.get("num_layers", 4),
                    d_state=preset_kwargs.get("d_state", 64),
                    d_conv=preset_kwargs.get("d_conv", 4),
                    expand=preset_kwargs.get("expand", 2.0),
                    dropout=preset_kwargs.get("dropout", 0.0),
                    norm_first=preset_kwargs.get("norm_first", True),
                )
        self.output_dim = latent_dimension

    def forward(self, vision_features: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if target_dtype is not None and vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        return self.resampler(vision_features)
