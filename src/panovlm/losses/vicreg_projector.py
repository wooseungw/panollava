"""Lightweight MLP projector used for VICReg training."""

from __future__ import annotations

import torch
import torch.nn as nn


class VICRegProjector(nn.Module):
    """Token-wise MLP projector for VICReg features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        hd = hidden_dim or max(in_dim, out_dim)
        layers: list[nn.Module] = []
        d_prev = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_prev, hd))
            layers.append(nn.LayerNorm(hd) if use_ln else nn.Identity())
            layers.append(nn.GELU())
            d_prev = hd
        layers.append(nn.Linear(d_prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BVS, S, D = x.shape
        x = x.reshape(-1, D)
        x = self.mlp(x)
        return x.view(BVS, S, -1)
