"""Utilities for computing VICReg overlap losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..utils import infer_hw


def compute_vicreg_overlap_loss(
    vision_output: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    overlap_ratio: float = 0.5,
    pair_chunk: Optional[int] = 8,
    vicreg_loss_module,
    has_cls_token: bool,
    vicreg_mode: str = "pairwise",
) -> torch.Tensor:
    if num_views <= 1:
        return torch.zeros((), device=vision_output.device)

    patch_features = vision_output[:, 1:] if has_cls_token else vision_output
    num_patches = patch_features.size(1)
    grid_h, grid_w = infer_hw(num_patches)
    D = patch_features.size(-1)
    patch_features = patch_features.view(batch_size, num_views, grid_h, grid_w, D)

    k = max(1, int(grid_w * overlap_ratio))
    curr = patch_features[:, :, :, -k:, :]
    nxt = torch.roll(patch_features, shifts=-1, dims=1)
    nxt = nxt[:, :, :, :k, :]

    P = batch_size * num_views
    L = grid_h * k
    curr = curr.contiguous().view(P, L, D)
    nxt = nxt.contiguous().view(P, L, D)

    inv_pair = F.mse_loss(curr, nxt, reduction='none').mean(dim=(1, 2))

    eps = 1e-4
    gamma = getattr(vicreg_loss_module, 'gamma', 1.0)

    # Variance calculation based on mode
    if vicreg_mode == "batchwise":
        # Original VICReg: compute variance across entire batch (dim=0)
        combined = torch.cat([curr, nxt], dim=0)  # [2P, L, D]
        combined_flat = combined.reshape(-1, D)   # [2P*L, D]
        std_combined = torch.sqrt(combined_flat.var(dim=0, unbiased=False) + eps)  # [D]
        var_term = F.relu(gamma - std_combined).mean()  # scalar
        var_pair = var_term.expand(P)  # broadcast to [P] for consistency
    else:
        # Pairwise mode (default): compute variance per pair (dim=1)
        std_x = torch.sqrt(curr.var(dim=1, unbiased=False) + eps)
        std_y = torch.sqrt(nxt.var(dim=1, unbiased=False) + eps)
        var_pair = 0.5 * (F.relu(gamma - std_x).mean(dim=1) + F.relu(gamma - std_y).mean(dim=1))

    # Covariance calculation based on mode
    if vicreg_mode == "batchwise":
        # Original VICReg: compute covariance across entire batch
        combined = torch.cat([curr, nxt], dim=0)  # [2P, L, D]
        combined_flat = combined.reshape(-1, D)   # [2P*L, D]
        combined_c = combined_flat - combined_flat.mean(dim=0, keepdim=True)
        denom_batch = max(combined_flat.size(0) - 1, 1)
        C = (combined_c.T @ combined_c) / denom_batch  # [D, D]
        C_offdiag = C.clone()
        C_offdiag.diagonal().zero_()
        cov_term = (C_offdiag ** 2).sum() / D  # scalar
        cov_pair = cov_term.expand(P)  # broadcast to [P] for consistency
    else:
        # Pairwise mode (default): compute covariance per pair
        curr_c = curr - curr.mean(dim=1, keepdim=True)
        nxt_c = nxt - nxt.mean(dim=1, keepdim=True)
        denom = max(L - 1, 1)

        def _cov_offdiag_sq_mean(xc: torch.Tensor) -> torch.Tensor:
            P_local = xc.size(0)
            if pair_chunk is None or pair_chunk >= P_local:
                C = torch.bmm(xc.transpose(1, 2), xc) / denom
                C2_sum = (C ** 2).sum(dim=(1, 2))
                diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                offdiag_sq = C2_sum - diag_sq
                return offdiag_sq / D
            out = []
            for s in range(0, P_local, pair_chunk):
                e = min(s + pair_chunk, P_local)
                C = torch.bmm(xc[s:e].transpose(1, 2), xc[s:e]) / denom
                C2_sum = (C ** 2).sum(dim=(1, 2))
                diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                offdiag_sq = C2_sum - diag_sq
                out.append(offdiag_sq / D)
                del C
            return torch.cat(out, dim=0)

        cov_x = _cov_offdiag_sq_mean(curr_c)
        cov_y = _cov_offdiag_sq_mean(nxt_c)
        cov_pair = 0.5 * (cov_x + cov_y)

    w_inv = getattr(vicreg_loss_module, 'similarity_weight', 25.0)
    w_var = getattr(vicreg_loss_module, 'variance_weight', 25.0)
    w_cov = getattr(vicreg_loss_module, 'covariance_weight', 1.0)

    per_pair = w_inv * inv_pair + w_var * var_pair + w_cov * cov_pair
    total = per_pair.mean()

    # Debug logging (can be disabled in production)
    if hasattr(vicreg_loss_module, '_debug_vicreg') and vicreg_loss_module._debug_vicreg:
        inv_mean = inv_pair.mean().item()
        var_mean = var_pair.mean().item() if isinstance(var_pair, torch.Tensor) else var_pair
        cov_mean = cov_pair.mean().item() if isinstance(cov_pair, torch.Tensor) else cov_pair
        print(f"[VICReg Debug] inv={inv_mean:.4f}, var={var_mean:.4f}, cov={cov_mean:.4f}, "
              f"weighted: inv={w_inv*inv_mean:.2f}, var={w_var*var_mean:.2f}, cov={w_cov*cov_mean:.2f}, "
              f"total={total.item():.2f}")

    return torch.clamp(total, max=1e6)
