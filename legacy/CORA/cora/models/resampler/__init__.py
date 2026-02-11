"""Resampler wrappers for CORA."""

from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn
from copy import deepcopy
import logging

from .resamplers import MLPResampler, QFormerResampler, IdentityResampler, AvgPoolResampler, ConvResampler

try:
    from .bimamba import BidirectionalMambaResampler
    _BIMAMBA_AVAILABLE = True
except ImportError:
    BidirectionalMambaResampler = None
    _BIMAMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

CANONICAL_RESAMPLER_TYPES = {
    "mlp": "mlp",
    "bimamba": "bimamba",
    "bidirectional_mamba": "bimamba",
    "bi_mamba": "bimamba",
    "qformer": "qformer",
    "identity": "identity",
    "avg": "avg",
    "conv": "conv",
}

class ResamplerModule(nn.Module):
    def __init__(self, config: Any, input_dim: int):
        super().__init__()
        
        # 1. Resolve Type
        resampler_type = getattr(config, 'resampler_type', 'mlp')
        canonical_type = CANONICAL_RESAMPLER_TYPES.get(resampler_type, resampler_type)
        
        # 2. Get Configs
        latent_dimension = int(getattr(config, 'latent_dimension', 768))
        resampler_cfg = getattr(config, 'resampler_config', {})
        if not isinstance(resampler_cfg, dict):
             # Handle case where pydantic model might be passed (though unlikely with proper schema usage)
             if hasattr(resampler_cfg, 'model_dump'):
                 resampler_cfg = resampler_cfg.model_dump(exclude_none=True)
             else:
                 resampler_cfg = {}

        logger.info(f"Initializing Resampler: {canonical_type} (in={input_dim}, out={latent_dimension})")
        logger.debug(f"Resampler Config: {resampler_cfg}")

        # 3. Instantiate
        if canonical_type == "mlp":
            self.resampler = MLPResampler(
                vision_dim=input_dim,
                latent_dim=latent_dimension,
                hidden_dim=resampler_cfg.get("hidden_dim"),
                depth=int(resampler_cfg.get("depth", 3)),
                use_ln=resampler_cfg.get("use_ln", True),
                pool_tokens=resampler_cfg.get("pool_tokens"),
                pool_type=str(resampler_cfg.get("pool_type", "avg")),
            )
        elif canonical_type == "bimamba":
            if not _BIMAMBA_AVAILABLE:
                logger.warning("BiMamba requested but not available. Falling back to MLP.")
                # Fallback to MLP with similar depth
                self.resampler = MLPResampler(
                    vision_dim=input_dim,
                    latent_dim=latent_dimension,
                    depth=int(resampler_cfg.get("depth", 3)),
                )
            else:
                 self.resampler = BidirectionalMambaResampler(
                    in_dim=input_dim,
                    out_dim=latent_dimension,
                    hidden_dim=resampler_cfg.get("hidden_dim"),
                    num_layers=int(resampler_cfg.get("depth", 4)),
                    d_state=int(resampler_cfg.get("d_state", 64)),
                    d_conv=int(resampler_cfg.get("d_conv", 4)),
                    expand=float(resampler_cfg.get("expand", 2.0)),
                    dropout=float(resampler_cfg.get("dropout", 0.0)),
                    norm_first=bool(resampler_cfg.get("norm_first", True)),
                )
        elif canonical_type == "qformer":
             self.resampler = QFormerResampler(
                in_dim=input_dim,
                out_dim=latent_dimension,
                num_query=int(resampler_cfg.get("num_query_tokens", 64)),
                num_hidden_layers=int(resampler_cfg.get("num_hidden_layers", 6)),
            )
        elif canonical_type == "conv":
            self.resampler = ConvResampler(
                in_dim=input_dim,
                out_dim=latent_dimension,
                num_tokens=int(resampler_cfg.get("num_tokens", 144)), # Unused in logic but might be for future
                depths=resampler_cfg.get("depths", [2, 2]),
                drop_path_rate=resampler_cfg.get("drop_path_rate", 0.0),
            )
        elif canonical_type == "identity":
            self.resampler = IdentityResampler(input_dim, latent_dimension)
        elif canonical_type == "avg":
            self.resampler = AvgPoolResampler(input_dim, latent_dimension)
        else:
            raise ValueError(f"Unsupported resampler type: {canonical_type}")
            
        self.output_dim = latent_dimension

    def forward(self, vision_features: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if target_dtype is not None and vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        return self.resampler(vision_features)

__all__ = ["ResamplerModule"]
