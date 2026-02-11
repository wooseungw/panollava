"""Panorama-aware projection and stitching helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import infer_hw
from .utils import resolve_module_dtype


class PanoramaProjector(nn.Module):
    def __init__(self, config: object, latent_dimension: int, language_hidden_size: int):
        super().__init__()
        self.use_projection_pe = bool(getattr(config, 'use_projection_positional_encoding', True))
        if self.use_projection_pe:
            from .positional import PanoramaPositionalEncoding

            try:
                vicreg_or_default = float(getattr(config, 'overlap_ratio', 0.5))
                pe_kwargs = dict(
                    embed_dim=latent_dimension,
                    view_encoding_type=getattr(config, 'pe_view_encoding_type', 'sinusoidal'),
                    spatial_encoding_type=getattr(config, 'pe_spatial_encoding_type', 'sinusoidal'),
                    enable_continuity=bool(getattr(config, 'pe_enable_continuity', True)),
                    overlap_ratio=vicreg_or_default,
                    temperature=float(getattr(config, 'pe_temperature', 10000.0)),
                    dropout=float(getattr(config, 'pe_dropout', 0.0)),
                )
            except Exception:
                pe_kwargs = dict(embed_dim=latent_dimension)
            self.projection_pe = PanoramaPositionalEncoding(**pe_kwargs)
        else:
            self.projection_pe = None

        self.linear = nn.Linear(latent_dimension, language_hidden_size)

        self.overlap_ratio = float(getattr(config, 'overlap_ratio', 0.5))
        self.stitching_mode = str(getattr(config, 'stitching_mode', 'stride_views'))
        self.stitch_stride_offset = int(getattr(config, 'stitch_stride_offset', 0))
        self.stitch_target_cols = int(getattr(config, 'stitch_target_cols', 0))
        self.stitch_target_to_view_width = bool(getattr(config, 'stitch_target_to_view_width', False))
        self.stitch_interp = str(getattr(config, 'stitch_interp', 'nearest'))

    def forward(
        self,
        resampled_features: torch.Tensor,
        batch_size: int,
        num_views: int,
        dtype_cache: dict[str, torch.dtype],
        language_model: nn.Module,
    ) -> torch.Tensor:
        proj_dtype = resolve_module_dtype(dtype_cache, "vision_to_language_projection", self.linear, resampled_features.dtype)
        if proj_dtype is not None and resampled_features.dtype != proj_dtype:
            resampled_features = resampled_features.to(proj_dtype)

        if self.use_projection_pe and self.projection_pe is not None:
            resampled_features = self.projection_pe(resampled_features, batch_size, num_views)
            if proj_dtype is not None and resampled_features.dtype != proj_dtype:
                resampled_features = resampled_features.to(proj_dtype)

        BV, S, D = resampled_features.shape
        try:
            H, W = infer_hw(S)
            x5 = resampled_features.view(batch_size, num_views, H, W, D)
            y5 = self.linear(x5)
            combined = self._prepare_combining(y5)
        except Exception:
            x = resampled_features.view(batch_size, num_views * S, D)
            combined = self.linear(x)

        lm_dtype = resolve_module_dtype(dtype_cache, "language_model", language_model, combined.dtype)
        if lm_dtype is not None and combined.dtype != lm_dtype:
            combined = combined.to(lm_dtype)
        return combined

    def _prepare_combining(self, projected_features: torch.Tensor) -> torch.Tensor:
        try:
            y5 = projected_features
            B, V, H, W, D = y5.shape
            ratio = self.overlap_ratio
            k = int(max(0, min(W, round(W * ratio))))

            mode = self.stitching_mode

            if mode == 'concat':
                y = y5.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
                return y.view(B, H * (V * W), D)

            if mode == 'resample':
                return self._resample_stitch(y5)

            if V <= 1 or k <= 0:
                y = y5.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
                return y.view(B, H * (V * W), D)

            parts = [y5[:, 0]]
            for v in range(1, V):
                parts.append(y5[:, v, :, k:, :])
            rowwise = torch.cat(parts, dim=2)
            return rowwise.reshape(B, -1, D)
        except Exception:
            B, V, H, W, D = projected_features.shape
            y = projected_features.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
            return y.view(B, H * (V * W), D)

    def _resample_stitch(self, projected_features: torch.Tensor) -> torch.Tensor:
        try:
            B, V, H, W, D = projected_features.shape
            ratio = self.overlap_ratio
            target_cols = self.stitch_target_cols
            k = int(max(0, min(W, round(W * ratio))))
            if target_cols <= 0:
                unique_cols = W * V - max(0, V - 1) * k
            else:
                unique_cols = target_cols

            s_float = float(W - k) if k > 0 else float(W)
            if s_float <= 0:
                s_float = float(W)

            if bool(self.stitch_target_to_view_width):
                T = int(W)
            else:
                T = unique_cols

            g = torch.linspace(0.0, float(unique_cols) - 1.0, steps=T, device=projected_features.device)
            v_idx = torch.arange(V, device=projected_features.device, dtype=torch.float32).view(V, 1)
            x_local = g.view(1, T) - (v_idx * s_float)

            xN = projected_features.permute(0, 1, 4, 2, 3).contiguous().view(B * V, D, H, W)

            if H > 1:
                y_lin = torch.linspace(-1.0, 1.0, steps=H, device=projected_features.device)
            else:
                y_lin = torch.zeros(1, device=projected_features.device)
            if W > 1:
                x_norm = (2.0 * (x_local / float(W - 1))) - 1.0
            else:
                x_norm = torch.zeros(V, T, device=projected_features.device)

            x_grid = x_norm.view(V, 1, T).expand(V, H, T)
            y_grid = y_lin.view(1, H, 1).expand(V, H, T)
            grid = torch.stack([x_grid, y_grid], dim=-1)
            grid = grid.unsqueeze(0).expand(B, V, H, T, 2).contiguous().view(B * V, H, T, 2)

            interp = str(self.stitch_interp).lower()
            gs_mode = 'bilinear' if interp == 'linear' else 'nearest'

            sampled = F.grid_sample(xN, grid, mode=gs_mode, padding_mode='zeros', align_corners=True)
            ones = torch.ones(B * V, 1, H, W, device=projected_features.device, dtype=projected_features.dtype)
            w = F.grid_sample(ones, grid, mode=gs_mode, padding_mode='zeros', align_corners=True)

            sampled = sampled.view(B, V, D, H, T).permute(0, 1, 3, 4, 2)
            w = w.view(B, V, 1, H, T).permute(0, 1, 3, 4, 2)
            num = (sampled * w).sum(dim=1)
            den = w.sum(dim=1).clamp_min(1e-6)
            fused = num / den
            return fused.reshape(B, -1, D)
        except Exception:
            B, V, H, W, D = projected_features.shape
            y = projected_features.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
            return y.view(B, H * (V * W), D)
