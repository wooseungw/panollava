# anyres_integration.py
# AnyRes ERP processor와 VICReg loss를 통합하는 모듈

from typing import List, Tuple, Dict, Any
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class TileRelationship:
    """타일 간의 공간 관계를 나타내는 클래스"""
    tile_idx1: int
    tile_idx2: int
    overlap_ratio: float
    yaw_distance: float
    pitch_distance: float
    relationship_type: str  # 'horizontal', 'vertical', 'diagonal', 'none'

class AnyResVICRegPairing:
    """
    AnyRes 타일 grid에서 VICReg loss를 위한 페어링 전략
    """

    def __init__(self,
                 hfov_deg: float = 90.0,
                 vfov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 pairing_strategy: str = "adjacent",  # 'adjacent' | 'distance_based' | 'all_pairs'
                 distance_threshold: float = 120.0):  # degrees
        """
        Args:
            hfov_deg: 타일의 수평 FOV
            vfov_deg: 타일의 수직 FOV
            overlap_ratio: 타일 간 겹침 비율 (anyres_e2p.py의 overlap과 동일)
            pairing_strategy: 페어링 전략
            distance_threshold: distance_based 전략에서 사용할 각도 임계값
        """
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.overlap_ratio = overlap_ratio
        self.pairing_strategy = pairing_strategy
        self.distance_threshold = distance_threshold

    @staticmethod
    def angular_distance(yaw1: float, pitch1: float, yaw2: float, pitch2: float) -> float:
        """
        구면 상의 두 점 사이의 각도 거리 계산 (Great Circle Distance)

        Returns:
            각도 거리 (degrees)
        """
        # Convert to radians
        y1, p1 = np.radians(yaw1), np.radians(pitch1)
        y2, p2 = np.radians(yaw2), np.radians(pitch2)

        # Haversine formula
        dlat = p2 - p1
        dlon = y2 - y1
        a = np.sin(dlat/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return np.degrees(c)

    @staticmethod
    def compute_overlap_area(yaw1: float, pitch1: float, hfov1: float, vfov1: float,
                            yaw2: float, pitch2: float, hfov2: float, vfov2: float) -> float:
        """
        두 타일 간의 겹침 영역 비율 추정

        Returns:
            겹침 비율 (0.0 ~ 1.0)
        """
        # 수평 방향 겹침
        yaw_diff = abs(yaw1 - yaw2)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff

        h_overlap = max(0, (hfov1 + hfov2) / 2 - yaw_diff) / min(hfov1, hfov2)

        # 수직 방향 겹침
        pitch_diff = abs(pitch1 - pitch2)
        v_overlap = max(0, (vfov1 + vfov2) / 2 - pitch_diff) / min(vfov1, vfov2)

        # 2D 겹침 비율
        overlap_ratio = h_overlap * v_overlap
        return float(np.clip(overlap_ratio, 0.0, 1.0))

    def build_tile_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """
        타일 메타데이터로부터 VICReg 페어 생성

        Args:
            tile_metas: anyres_e2p.build_anyres_from_erp()에서 반환된 metas
                       각 meta는 {'tile_id', 'yaw_deg', 'pitch_deg', 'hfov_deg', 'vfov_deg', ...}

        Returns:
            List of (idx1, idx2, overlap_ratio) tuples
        """
        if self.pairing_strategy == "adjacent":
            return self._build_adjacent_pairs(tile_metas)
        elif self.pairing_strategy == "distance_based":
            return self._build_distance_pairs(tile_metas)
        elif self.pairing_strategy == "all_pairs":
            return self._build_all_pairs(tile_metas)
        else:
            raise ValueError(f"Unknown pairing strategy: {self.pairing_strategy}")

    def _build_adjacent_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """
        Grid 상에서 인접한 타일만 페어링 (권장)
        - 수평 인접: yaw가 비슷하고 pitch가 한 단계 차이
        - 수직 인접: pitch가 비슷하고 yaw가 한 단계 차이
        """
        pairs = []
        n = len(tile_metas)

        # Organize tiles by (pitch, yaw) for grid structure
        pitch_groups: Dict[float, List[int]] = {}
        for i, meta in enumerate(tile_metas):
            pitch = round(meta['pitch_deg'], 1)  # 소수점 1자리로 그룹화
            if pitch not in pitch_groups:
                pitch_groups[pitch] = []
            pitch_groups[pitch].append(i)

        # 각 pitch 레벨에서 yaw로 정렬된 타일들을 수평으로 연결
        for pitch, indices in pitch_groups.items():
            # yaw로 정렬
            sorted_indices = sorted(indices, key=lambda i: tile_metas[i]['yaw_deg'])

            # 수평 인접 페어
            for j in range(len(sorted_indices) - 1):
                idx1, idx2 = sorted_indices[j], sorted_indices[j + 1]
                overlap = self.compute_overlap_area(
                    tile_metas[idx1]['yaw_deg'], tile_metas[idx1]['pitch_deg'],
                    tile_metas[idx1]['hfov_deg'], tile_metas[idx1]['vfov_deg'],
                    tile_metas[idx2]['yaw_deg'], tile_metas[idx2]['pitch_deg'],
                    tile_metas[idx2]['hfov_deg'], tile_metas[idx2]['vfov_deg']
                )
                if overlap > 0.01:  # 최소 1% 이상 겹칠 때만
                    pairs.append((idx1, idx2, overlap))

            # 순환 연결 (마지막 타일과 첫 타일 - 파노라마는 360도 닫힌 구조)
            if len(sorted_indices) > 2:
                idx1, idx2 = sorted_indices[-1], sorted_indices[0]
                yaw_diff = abs(tile_metas[idx1]['yaw_deg'] - tile_metas[idx2]['yaw_deg'])
                if yaw_diff > 180:  # 실제로 인접한지 확인 (예: -170도와 +170도)
                    overlap = self.compute_overlap_area(
                        tile_metas[idx1]['yaw_deg'], tile_metas[idx1]['pitch_deg'],
                        tile_metas[idx1]['hfov_deg'], tile_metas[idx1]['vfov_deg'],
                        tile_metas[idx2]['yaw_deg'], tile_metas[idx2]['pitch_deg'],
                        tile_metas[idx2]['hfov_deg'], tile_metas[idx2]['vfov_deg']
                    )
                    if overlap > 0.01:
                        pairs.append((idx1, idx2, overlap))

        # 수직 인접 페어 (다른 pitch 레벨 간)
        pitch_levels = sorted(pitch_groups.keys())
        for i in range(len(pitch_levels) - 1):
            pitch1, pitch2 = pitch_levels[i], pitch_levels[i + 1]
            indices1, indices2 = pitch_groups[pitch1], pitch_groups[pitch2]

            # 각 레벨의 타일들을 yaw 기준으로 매칭
            for idx1 in indices1:
                for idx2 in indices2:
                    # yaw가 비슷한 타일끼리만
                    yaw_diff = abs(tile_metas[idx1]['yaw_deg'] - tile_metas[idx2]['yaw_deg'])
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff

                    if yaw_diff < self.hfov_deg:  # FOV 범위 내
                        overlap = self.compute_overlap_area(
                            tile_metas[idx1]['yaw_deg'], tile_metas[idx1]['pitch_deg'],
                            tile_metas[idx1]['hfov_deg'], tile_metas[idx1]['vfov_deg'],
                            tile_metas[idx2]['yaw_deg'], tile_metas[idx2]['pitch_deg'],
                            tile_metas[idx2]['hfov_deg'], tile_metas[idx2]['vfov_deg']
                        )
                        if overlap > 0.01:
                            pairs.append((idx1, idx2, overlap))

        return pairs

    def _build_distance_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """각도 거리 기반 페어링"""
        pairs = []
        n = len(tile_metas)

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.angular_distance(
                    tile_metas[i]['yaw_deg'], tile_metas[i]['pitch_deg'],
                    tile_metas[j]['yaw_deg'], tile_metas[j]['pitch_deg']
                )

                if dist < self.distance_threshold:
                    overlap = self.compute_overlap_area(
                        tile_metas[i]['yaw_deg'], tile_metas[i]['pitch_deg'],
                        tile_metas[i]['hfov_deg'], tile_metas[i]['vfov_deg'],
                        tile_metas[j]['yaw_deg'], tile_metas[j]['pitch_deg'],
                        tile_metas[j]['hfov_deg'], tile_metas[j]['vfov_deg']
                    )
                    if overlap > 0.01:
                        pairs.append((i, j, overlap))

        return pairs

    def _build_all_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """모든 가능한 페어 (메모리 집약적)"""
        pairs = []
        n = len(tile_metas)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = self.compute_overlap_area(
                    tile_metas[i]['yaw_deg'], tile_metas[i]['pitch_deg'],
                    tile_metas[i]['hfov_deg'], tile_metas[i]['vfov_deg'],
                    tile_metas[j]['yaw_deg'], tile_metas[j]['pitch_deg'],
                    tile_metas[j]['hfov_deg'], tile_metas[j]['vfov_deg']
                )
                if overlap > 0.01:
                    pairs.append((i, j, overlap))

        return pairs


def compute_vicreg_anyres_loss(
    vision_features: torch.Tensor,  # [B*V, S, D]
    batch_size: int,
    tile_metas_batch: List[List[Dict[str, Any]]],  # List of tile_metas for each batch
    vicreg_loss_module,
    pairing_strategy: str = "adjacent",
    hfov_deg: float = 90.0,
    pair_chunk: int = 8,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """
    AnyRes 타일에 대한 VICReg loss 계산

    기존 compute_vicreg_overlap_loss와 유사하지만,
    순차적 인접 뷰 대신 2D grid 공간 인접 페어링 사용

    Args:
        vision_features: [B*V, S, D] - VICReg projector 출력
        batch_size: 배치 크기
        tile_metas_batch: 각 배치 샘플의 타일 메타데이터 리스트
        vicreg_loss_module: VicRegLoss 인스턴스
        pairing_strategy: 페어링 전략
        hfov_deg: 타일 FOV (anyres_e2p 설정과 동일)
        pair_chunk: 메모리 절약용 청킹
        has_cls_token: CLS 토큰 제외 여부

    Returns:
        VICReg loss (scalar)
    """
    import torch.nn.functional as F

    num_views = vision_features.size(0) // batch_size

    # CLS token 제외
    patch_features = vision_features[:, 1:] if has_cls_token else vision_features
    num_patches = patch_features.size(1)
    D = patch_features.size(-1)

    # Grid dimensions 추론
    grid_h = int(np.sqrt(num_patches))
    grid_w = num_patches // grid_h

    # [B*V, S, D] → [B, V, H, W, D]
    patch_features_grid = patch_features.view(batch_size, num_views, grid_h, grid_w, D)

    # 모든 페어 수집 및 처리
    all_curr_feats = []
    all_next_feats = []

    for b in range(batch_size):
        tile_metas = tile_metas_batch[b]

        # 이 배치 샘플의 타일 페어 생성
        pairing = AnyResVICRegPairing(
            hfov_deg=hfov_deg,
            vfov_deg=hfov_deg,
            overlap_ratio=0.5,
            pairing_strategy=pairing_strategy
        )

        pairs = pairing.build_tile_pairs(tile_metas)

        if not pairs:
            continue

        # 각 페어에 대해 feature 추출
        for idx1, idx2, overlap_ratio in pairs:
            # Global view 제외
            if tile_metas[idx1]['tile_id'] == 0 or tile_metas[idx2]['tile_id'] == 0:
                continue

            # 타일의 feature 가져오기 [H, W, D]
            feat1_grid = patch_features_grid[b, idx1]  # [H, W, D]
            feat2_grid = patch_features_grid[b, idx2]  # [H, W, D]

            # Overlap 영역 추출
            # overlap_ratio에 따라 겹치는 부분 선택
            if overlap_ratio > 0.1:  # 최소 10% 이상 겹칠 때만
                # 간단히 중앙 영역 사용 (정확한 geometric overlap은 복잡)
                k = max(1, int(grid_w * overlap_ratio))

                # feat1의 오른쪽 k 픽셀 (임의 선택 - 실제론 타일 위치에 따라 다름)
                feat1_overlap = feat1_grid[:, -k:, :]  # [H, k, D]
                # feat2의 왼쪽 k 픽셀
                feat2_overlap = feat2_grid[:, :k, :]   # [H, k, D]

                # Flatten to [L, D] (2D 구조 유지하되 batch용으로 flatten)
                feat1_flat = feat1_overlap.reshape(-1, D)  # [H*k, D]
                feat2_flat = feat2_overlap.reshape(-1, D)  # [H*k, D]

                all_curr_feats.append(feat1_flat)
                all_next_feats.append(feat2_flat)

    if len(all_curr_feats) == 0:
        return torch.zeros((), device=vision_features.device)

    # 모든 페어를 배치로 처리 [P, L, D]
    curr = torch.stack(all_curr_feats, dim=0)  # [P, L, D]
    nxt = torch.stack(all_next_feats, dim=0)   # [P, L, D]

    P = curr.size(0)
    L = curr.size(1)

    # VICReg 계산 (compute_vicreg_overlap_loss와 동일한 방식)
    eps = 1e-4
    gamma = getattr(vicreg_loss_module, 'gamma', 1.0)
    w_inv = getattr(vicreg_loss_module, 'similarity_weight', 25.0)
    w_var = getattr(vicreg_loss_module, 'variance_weight', 25.0)
    w_cov = getattr(vicreg_loss_module, 'covariance_weight', 1.0)

    # 1. Invariance (similarity)
    inv_pair = F.mse_loss(curr, nxt, reduction='none').mean(dim=(1, 2))  # [P]

    # 2. Variance (pairwise mode - default)
    std_x = torch.sqrt(curr.var(dim=1, unbiased=False) + eps)  # [P, D]
    std_y = torch.sqrt(nxt.var(dim=1, unbiased=False) + eps)   # [P, D]
    var_pair = 0.5 * (F.relu(gamma - std_x).mean(dim=1) + F.relu(gamma - std_y).mean(dim=1))  # [P]

    # 3. Covariance (pairwise mode)
    curr_c = curr - curr.mean(dim=1, keepdim=True)  # [P, L, D]
    nxt_c = nxt - nxt.mean(dim=1, keepdim=True)     # [P, L, D]
    denom = max(L - 1, 1)

    def _cov_offdiag_sq_mean(xc: torch.Tensor) -> torch.Tensor:
        P_local = xc.size(0)
        if pair_chunk is None or pair_chunk >= P_local:
            C = torch.bmm(xc.transpose(1, 2), xc) / denom  # [P, D, D]
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
    cov_pair = 0.5 * (cov_x + cov_y)  # [P]

    # Weighted sum
    per_pair = w_inv * inv_pair + w_var * var_pair + w_cov * cov_pair  # [P]
    total = per_pair.mean()

    return torch.clamp(total, max=1e6)
