"""Loss functions for CORA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

from cora.utils import infer_hw

logger = logging.getLogger(__name__)

class VICRegLoss(nn.Module):
    """
    VICReg Loss adapted for Panoramic Overlaps.
    Computes loss based on overlap consistency between adjacent views.
    """
    def __init__(
        self,
        similarity_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        gamma: float = 1.0,
        overlap_ratio: float = 0.5,
        vicreg_mode: str = "pairwise", # 'pairwise' or 'batchwise'
        eps: float = 1e-4,
        debug: bool = False
    ):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.gamma = gamma
        self.overlap_ratio = overlap_ratio
        self.vicreg_mode = vicreg_mode
        self.eps = eps
        self._debug = debug

    def forward(
        self,
        vision_features: torch.Tensor, # [BV, S, D]
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """
        Compute VICReg loss on vision features.
        Assumes vision features are from projector output.
        """
        if num_views <= 1:
            return vision_features.sum() * 0.0

        # 1. Infer Grid and Channels
        # vision_features: [BV, S, D]
        # We assume [CLS] token is handled or not present (Projector usually maps all tokens)
        # If input has CLS and we want to skip it, we should do it before or handle checks.
        # Here we assume vision_features are patch tokens only (or Projector output preserves shape).
        # Standard VisionBackbone output preserves [CLS] if present? 
        # CORA's VisionBackbone currently returns global pool or all tokens? 
        # Resampler usually takes all tokens. VICRegProjector projects resampler output.
        # Resampler output usually has no CLS unless explicitly added.
        
        BV, S, D = vision_features.shape
        grid_h, grid_w = infer_hw(S)
        
        # Reshape to [B, V, H, W, D]
        features_grid = vision_features.view(batch_size, num_views, grid_h, grid_w, D)
        
        # 2. Extract Overlaps
        k = max(1, int(grid_w * self.overlap_ratio))
        
        # Current view's right side (overlap region)
        curr_overlap = features_grid[:, :, :, -k:, :] # [B, V, H, k, D]
        
        # Next view's left side (overlap region)
        # Roll V dimension by -1 (next view)
        next_features = torch.roll(features_grid, shifts=-1, dims=1)
        next_overlap = next_features[:, :, :, :k, :] # [B, V, H, k, D]
        
        # 3. Flatten for calculation
        # We compare (Current View i, Next View i+1) pairs.
        # Total pairs P = B * V
        # Feature vector size L = H * k
        P = batch_size * num_views
        L = grid_h * k
        
        curr_flat = curr_overlap.contiguous().view(P, L, D)
        next_flat = next_overlap.contiguous().view(P, L, D)
        
        # 4. Invariance Loss (MSE)
        inv_loss = F.mse_loss(curr_flat, next_flat, reduction='none').mean(dim=(1, 2)) # [P]
        
        # 5. Variance Loss
        if self.vicreg_mode == "batchwise":
            # Original VICReg mode: variance across entire batch
            combined = torch.cat([curr_flat, next_flat], dim=0) # [2P, L, D]
            combined = combined.reshape(-1, D) # [2P*L, D]
            std = torch.sqrt(combined.var(dim=0, unbiased=False) + self.eps)
            var_term = F.relu(self.gamma - std).mean()
            var_loss = var_term.expand(P)
        else:
            # Pairwise mode (default for panorama): variance per pair logic
            # Actually, standard VICReg minimizes variance *within* a batch to prevent collapse.
            # "Pairwise" might mean variance calculation scope is limited?
            # Following legacy logic:
            std_curr = torch.sqrt(curr_flat.var(dim=1, unbiased=False) + self.eps) # [P, D]
            std_next = torch.sqrt(next_flat.var(dim=1, unbiased=False) + self.eps) # [P, D]
            var_loss = 0.5 * (F.relu(self.gamma - std_curr).mean(dim=1) + F.relu(self.gamma - std_next).mean(dim=1)) # [P]
            
        # 6. Covariance Loss
        curr_centered = curr_flat - curr_flat.mean(dim=1, keepdim=True)
        next_centered = next_flat - next_flat.mean(dim=1, keepdim=True)
        denom = max(L - 1, 1)
        
        # Compute off-diagonal covariance sum per pair
        # cov matrices [P, D, D] can be large if D is large.
        # We can sum squares directly without forming full matrix if memory is tight?
        # Legacy code did chunking. Let's do direct batch matmul if possible, or simple loop if safe.
        # With D=768 or 2048, [D, D] is small (4MB). P=Batch*Views (e.g. 4*8=32). 
        # 32 * 4MB is fine.
        
        cov_curr = torch.bmm(curr_centered.transpose(1, 2), curr_centered) / denom # [P, D, D]
        cov_next = torch.bmm(next_centered.transpose(1, 2), next_centered) / denom # [P, D, D]
        
        def off_diag_sum(cov):
            diag = torch.diagonal(cov, dim1=1, dim2=2)
            return (cov.pow(2).sum(dim=(1, 2)) - diag.pow(2).sum(dim=1)) / D
            
        cov_loss = 0.5 * (off_diag_sum(cov_curr) + off_diag_sum(cov_next)) # [P]
        
        # 7. Total Loss
        # Broadcast weights if necessary (they are scalars)
        # inv_loss: [P], var_loss: [P], cov_loss: [P]
        
        per_pair_loss = (
            self.similarity_weight * inv_loss +
            self.variance_weight * var_loss +
            self.covariance_weight * cov_loss
        )
        
        total_loss = per_pair_loss.mean()
        
        if self._debug:
            logger.info(f"VICReg: Inv={inv_loss.mean():.4f} Var={var_loss.mean():.4f} Cov={cov_loss.mean():.4f} Total={total_loss:.4f}")
            
        return total_loss
