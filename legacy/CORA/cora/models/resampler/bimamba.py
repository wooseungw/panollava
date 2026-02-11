"""Bidirectional Mamba-based resampler blocks."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .resamplers import BaseResampler

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None


def _ensure_mamba_available():
    """Ensure mamba_ssm is available."""
    if not MAMBA_AVAILABLE:
        raise ImportError(
            "mamba_ssm is not installed. Please install it with:\n"
            "  pip install mamba-ssm causal-conv1d\n"
            "Note: This requires CUDA to be properly configured."
        )


class _BidirectionalMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        d_state: int,
        d_conv: int,
        expand: float,
        dropout: float,
        norm_first: bool,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.forward_block = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_block = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        working = self.norm(x) if self.norm_first else x

        forward_out = self.forward_block(working)
        backward_out = torch.flip(working, dims=[1])
        backward_out = self.backward_block(backward_out)
        backward_out = torch.flip(backward_out, dims=[1])

        combined = 0.5 * (forward_out + backward_out)
        combined = self.dropout(combined)
        out = residual + combined

        if not self.norm_first:
            out = self.norm(out)
        return out


class BidirectionalMambaResampler(BaseResampler):
    """Resampler that processes tokens with bidirectional Mamba blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: float = 2.0,
        dropout: float = 0.0,
        norm_first: bool = True,
    ) -> None:
        self.hidden_dim = hidden_dim or out_dim
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.norm_first = norm_first
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        _ensure_mamba_available()

        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _BidirectionalMambaBlock(
                    self.hidden_dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.dropout,
                    norm_first=self.norm_first,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x

    @property
    def config(self):
        base_cfg = super().config
        base_cfg.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "d_state": self.d_state,
                "d_conv": self.d_conv,
                "expand": self.expand,
                "dropout": self.dropout,
                "norm_first": self.norm_first,
            }
        )
        return base_cfg


__all__ = ["BidirectionalMambaResampler"]
