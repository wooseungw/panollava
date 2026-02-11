import torch
import torch.nn as nn
import torch.nn.functional as F


class VicRegLoss(nn.Module):
    """
    VICReg loss with invariance, variance, and covariance terms.

    inv: MSE(x, y)
    var: mean(ReLU(gamma - std)) for x,y (½ 합)
    cov: off-diagonal^2 평균 (x,y 각각 ½), 분모 D로 정규화
    """
    def __init__(self, similarity_weight=25.0, variance_weight=25.0,
                 covariance_weight=1.0, gamma=1.0, use_ddp_gather=False):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.gamma = gamma
        self.use_ddp_gather = use_ddp_gather

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:]

    def _gather_if_needed(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_ddp_gather:
            return z
        # 필요 시 all_gather로 확장 가능
        return z

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        if N < 2:
            return F.mse_loss(x, y) * self.similarity_weight

        inv = F.mse_loss(x, y)

        xg = self._gather_if_needed(x)
        yg = self._gather_if_needed(y)

        std_x = torch.sqrt(xg.var(dim=0, unbiased=False) + 1e-4)
        std_y = torch.sqrt(yg.var(dim=0, unbiased=False) + 1e-4)
        var = 0.5 * (F.relu(self.gamma - std_x).mean() + F.relu(self.gamma - std_y).mean())

        x_c = xg - xg.mean(dim=0, keepdim=True)
        y_c = yg - yg.mean(dim=0, keepdim=True)
        denom = max(xg.size(0) - 1, 1)
        cov_x = (x_c.T @ x_c) / denom
        cov_y = (y_c.T @ y_c) / denom
        cov = 0.5 * (
            self._off_diagonal(cov_x).pow(2).sum() / D
            + self._off_diagonal(cov_y).pow(2).sum() / D
        )

        total = (
            self.similarity_weight * inv
            + self.variance_weight * var
            + self.covariance_weight * cov
        )
        if not torch.isfinite(total):
            total = torch.zeros((), device=x.device, dtype=x.dtype)
        return total


from .vicreg_overlap import compute_vicreg_overlap_loss
from .vicreg_projector import VICRegProjector

__all__ = [
    "VicRegLoss",
    "compute_vicreg_overlap_loss",
    "VICRegProjector",
]
