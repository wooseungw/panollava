"""Panorama-aware positional encoding modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..utils import infer_hw


class PanoramaPositionalEncoding(nn.Module):
    """Panorama-aware positional encoding enforcing yaw continuity across views."""

    def __init__(
        self,
        *,
        embed_dim: int,
        view_encoding_type: str = "sinusoidal",
        spatial_encoding_type: str = "sinusoidal",
        enable_continuity: bool = True,
        overlap_ratio: float = 0.0,
        temperature: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.view_encoding_type = view_encoding_type
        self.spatial_encoding_type = spatial_encoding_type
        self.enable_continuity = bool(enable_continuity)
        self.temperature = float(temperature)
        self.overlap_ratio = max(0.0, min(float(overlap_ratio), 0.999))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    @staticmethod
    def _build_sinusoidal(pos: torch.Tensor, dim: int, temperature: float, dtype: torch.dtype) -> torch.Tensor:
        device = pos.device
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        pos_f = pos.to(compute_dtype)
        half = dim // 2
        if half == 0:
            return torch.zeros(*pos.shape, dim, device=device, dtype=dtype)
        idx = torch.arange(half, device=device, dtype=compute_dtype)
        div = torch.exp(-math.log(temperature) * (2 * idx / max(1, dim)))
        ang = pos_f.unsqueeze(-1) * div
        emb_sin = torch.sin(ang)
        emb_cos = torch.cos(ang)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        if emb.shape[-1] < dim:
            pad = torch.zeros(*emb.shape[:-1], dim - emb.shape[-1], device=device, dtype=compute_dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb.to(dtype)

    def _yaw_encoding(
        self,
        num_views: int,
        batch_size: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        V, D = num_views, self.embed_dim
        s = 1.0 - self.overlap_ratio
        if not self.enable_continuity:
            s = 1.0
        base_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        v_idx = torch.arange(V, device=device, dtype=base_dtype).view(V, 1)
        x = torch.arange(W, device=device, dtype=base_dtype) / max(1.0, float(W))
        g = v_idx * s + x.unsqueeze(0)
        L_total = V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)
        phi = (2.0 * math.pi) * (g / max(1e-6, float(L_total)))
        yaw_vw = self._build_sinusoidal(phi, D, self.temperature, base_dtype)
        yaw = yaw_vw.view(1, V, 1, W, D).expand(batch_size, V, H, W, D)
        return yaw.to(dtype)

    def _spatial_encoding(self, H: int, W: int, V: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.spatial_encoding_type != "sinusoidal":
            return torch.zeros(V, H, W, self.embed_dim, device=device, dtype=dtype)
        base_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        y = torch.arange(H, device=device, dtype=base_dtype)
        y_emb = self._build_sinusoidal(y, self.embed_dim, self.temperature, base_dtype)
        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        v_idx = torch.arange(V, device=device, dtype=base_dtype).view(V, 1)
        x_local = torch.arange(W, device=device, dtype=base_dtype) / max(1.0, float(W))
        g = v_idx * s + x_local.unsqueeze(0)
        x_emb = self._build_sinusoidal(g, self.embed_dim, self.temperature, base_dtype)
        grid = y_emb.view(1, H, 1, self.embed_dim) + x_emb.view(V, 1, W, self.embed_dim)
        return grid.to(dtype)

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        BV, S, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Embed dim mismatch: x={D}, pe={self.embed_dim}")
        H, W = infer_hw(S)
        xv = x.view(batch_size, num_views, H, W, D)
        yaw_pe = self._yaw_encoding(num_views, batch_size, H, W, x.device, x.dtype)
        spat_pe = self._spatial_encoding(H, W, num_views, x.device, x.dtype)
        pe = yaw_pe + spat_pe
        out = (xv + pe).view(BV, S, D)
        return self.dropout(out)


class PanoramaPositionalEncoding2(nn.Module):
    """Spherical 3D positional encoding for panoramic tokens."""

    def __init__(
        self,
        *,
        embed_dim: int,
        view_encoding_type: str = "sinusoidal",
        spatial_encoding_type: str = "sinusoidal",
        enable_continuity: bool = True,
        overlap_ratio: float = 0.0,
        temperature: float = 10000.0,
        dropout: float = 0.0,
        num_fourier_bands: int = 8,
        include_input_xyz: bool = True,
        pe_scale: float = math.pi,
        phi_offset_rad: float = 0.0,
        lat_center_rad: float = 0.0,
        lat_coverage_ratio: float = 1.0,
        project_bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.enable_continuity = bool(enable_continuity)
        self.overlap_ratio = float(max(0.0, min(overlap_ratio, 0.999)))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.num_fourier_bands = int(num_fourier_bands)
        self.include_input_xyz = bool(include_input_xyz)
        self.pe_scale = float(pe_scale)
        self.phi_offset_rad = float(phi_offset_rad)
        self.lat_center_rad = float(lat_center_rad)
        self.lat_coverage_ratio = float(lat_coverage_ratio)
        raw_dim = self._raw_feature_dim()
        self._proj = nn.Linear(raw_dim, embed_dim, bias=project_bias)

    def _raw_feature_dim(self) -> int:
        xyz_dim = 3 if self.include_input_xyz else 0
        return xyz_dim + (3 * (2 * self.num_fourier_bands))

    def _infer_hw(self, tokens: int) -> tuple[int, int]:
        return infer_hw(tokens)

    def _global_longitude(self, V: int, W: int, device: torch.device) -> torch.Tensor:
        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        v_idx = torch.arange(V, device=device, dtype=torch.float32).view(V, 1)
        x = torch.arange(W, device=device, dtype=torch.float32) / max(1.0, float(W))
        g = v_idx * s + x.unsqueeze(0)
        L_total = V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)
        L_total = max(float(L_total), 1e-6)
        phi = (2.0 * math.pi) * (g / L_total) + self.phi_offset_rad
        return phi

    def _latitude_from_rows(self, H: int, device: torch.device) -> torch.Tensor:
        y = torch.arange(H, device=device, dtype=torch.float32)
        u = (y + 0.5) / max(1.0, float(H))
        theta_raw = (u - 0.5) * math.pi
        theta = self.lat_center_rad + (self.lat_coverage_ratio * theta_raw)
        return theta

    @staticmethod
    def _spherical_xyz(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(theta)
        x = cos_theta * torch.cos(phi)
        y = torch.sin(theta)
        z = cos_theta * torch.sin(phi)
        return torch.stack([x, y, z], dim=-1)

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        bands = self.num_fourier_bands
        if bands <= 0:
            raise ValueError("num_fourier_bands must be positive")
        freq = (2.0 ** torch.arange(bands, device=coords.device, dtype=coords.dtype)) * self.pe_scale
        ang = coords.unsqueeze(-1) * freq
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        out = torch.cat([sin, cos], dim=-1)
        return out.view(*coords.shape[:-1], 3 * (2 * bands))

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        BV, S, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Embed dim mismatch: x={D}, pe={self.embed_dim}")
        H, W = self._infer_hw(S)
        device = x.device
        xv = x.view(batch_size, num_views, H, W, D)
        phi_vw = self._global_longitude(num_views, W, device)
        theta_h = self._latitude_from_rows(H, device)
        phi = phi_vw.view(1, num_views, 1, W).expand(batch_size, num_views, H, W)
        theta = theta_h.view(1, 1, H, 1).expand(batch_size, num_views, H, W)
        xyz = self._spherical_xyz(theta, phi)
        feats = []
        if self.include_input_xyz:
            feats.append(xyz)
        feats.append(self._fourier_encode(xyz))
        pe_raw = torch.cat(feats, dim=-1)
        pe = self._proj(pe_raw)
        out = xv + pe
        out = out.view(BV, S, D)
        return self.dropout(out)
