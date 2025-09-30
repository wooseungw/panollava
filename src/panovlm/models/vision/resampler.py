"""Resampler wrappers for panoramic features."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..resampler.resamplers import MLPResampler


class ResamplerModule(nn.Module):
    def __init__(self, config: Any, input_dim: int):
        super().__init__()
        resampler_type = getattr(config, 'resampler_type', 'mlp')
        latent_dimension = int(getattr(config, 'latent_dimension', input_dim))
        resampler_depth = getattr(config, 'resampler_depth', 2)
        resampler_hidden_dim = getattr(config, 'resampler_hidden_dim', None)
        resampler_use_ln = getattr(config, 'resampler_use_ln', True)

        if resampler_type == "mlp":
            self.resampler = MLPResampler(
                input_dim,
                latent_dimension,
                hidden_dim=resampler_hidden_dim,
                depth=resampler_depth,
                use_ln=resampler_use_ln,
            )
        else:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {resampler_type}")
        self.output_dim = latent_dimension

    def forward(self, vision_features: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if target_dtype is not None and vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        return self.resampler(vision_features)
