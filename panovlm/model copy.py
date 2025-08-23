# coding: utf-8

import math
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# LoRA 지원을 위한 PEFT import (선택적)
try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA fine-tuning will not be supported.")

# ==================== Reproducibility Utility ====================
import os, random, numpy as np

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """전역 재현성(가능한 한)을 위한 시드 고정 유틸리티.
    Args:
        seed: 시드 값
        deterministic: True이면 cudnn deterministic 모드 활성 (성능 ↓ 가능)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        else:
            torch.backends.cudnn.benchmark = True      # 성능 우선
    except Exception as _e:  # torch 미존재 환경 대비
        print(f"[set_seed] Torch seed setup skipped: {_e}")

def _safe_save_pretrained(model, save_path: str, **kwargs) -> bool:
    """
    HuggingFace 모델을 안전하게 저장하는 유틸리티 함수
    
    최근 HuggingFace Hub 업데이트로 인한 repo_id 검증 문제를 해결합니다.
    """
    if not hasattr(model, 'save_pretrained'):
        return False
    
    # 안전한 기본 kwargs 설정
    safe_kwargs = {
        'push_to_hub': False,
        'token': False,
        'safe_serialization': kwargs.get('safe_serialization', True),
        **kwargs
    }
    
    # repo_id, from_id, to_id 등 Hub 관련 파라미터 제거
    hub_params = ['repo_id', 'from_id', 'to_id', 'hub_model_id']
    for param in hub_params:
        safe_kwargs.pop(param, None)
    
    try:
        model.save_pretrained(save_path, **safe_kwargs)
        return True
    except Exception as e:
        print(f"Warning: Failed to save with SafeTensors: {e}")
        # Fallback to PyTorch format
        try:
            fallback_kwargs = {k: v for k, v in safe_kwargs.items() if k != 'safe_serialization'}
            model.save_pretrained(save_path, **fallback_kwargs)
            return True
        except Exception as e2:
            print(f"Error: Failed to save model completely: {e2}")
            return False

def _infer_hw(num_patches: int) -> tuple[int, int]:
    """
    주어진 패치 토큰 개수(`num_patches`)로부터 (H, W) 그리드 크기를 추정합니다.
    1) 완전제곱 → 정사각형  
    2) 아니면 √N 이하에서 가장 큰 약수를 찾아 (height, width = N//height) 반환  
       (예: 256 → 16×16, 140 → 10×14)
    """
    height = int(math.sqrt(num_patches))
    if height * height == num_patches:
        return height, height

    for height in range(height, 0, -1):
        if num_patches % height == 0:
            return height, num_patches // height
    raise ValueError(f"그리드 추정 실패: 패치 수={num_patches}")
# ---------------------------------------------------------------------------
# ‣ Resampler
# ---------------------------------------------------------------------------
class MLPResampler(nn.Module):
    """다층 퍼셉트론을 사용한 리샘플러
    입력: (배치, 뷰 * 패치수, 비전차원) → 출력: (배치, 뷰 * 패치수, 잠재차원)
    """

    def __init__(self, vision_dim: int, latent_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_dim, latent_dim)
        self.bn1 = nn.LayerNorm(latent_dim, eps=1e-5)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # vision_features: (batch, seq, vision_dim) 또는 (N, vision_dim)
        orig_shape = vision_features.shape
        if vision_features.dim() == 3:
            batch, seq, dim = vision_features.shape
            x = vision_features.reshape(-1, dim)  # (batch*seq, dim)
        else:
            x = vision_features
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.linear2(x)
        if vision_features.dim() == 3:
            x = x.view(orig_shape[0], orig_shape[1], -1)
        return x

# ---------------------------------------------------------------------------
# ‣ Geometric Transformation Utilities for VICReg-L
# ---------------------------------------------------------------------------

def create_erp_to_pinhole_grid(height: int, width: int, metadata: dict) -> torch.Tensor:
    """
    ERP 좌표계에서 pinhole 좌표계로의 backward warping grid 생성
    
    Args:
        height, width: 특징맵 해상도
        metadata: {'yaw': float, 'pitch': float, 'effective_fov': float}
    
    Returns:
        grid: [1, H, W, 2] backward grid for F.grid_sample
    """
    
    yaw_deg = metadata['yaw']
    pitch_deg = metadata.get('pitch', 0.0)
    fov_deg = metadata['effective_fov']
    
    # Pinhole 좌표 생성 (타겟 좌표계)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height), 
        torch.linspace(-1, 1, width), 
        indexing='ij'
    )
    
    # Pinhole unproject: normalized coords -> 3D direction
    fov_rad = math.radians(fov_deg)
    tan_half_fov = math.tan(fov_rad / 2)
    
    X = tan_half_fov * x  # horizontal direction
    Y = tan_half_fov * y  # vertical direction  
    Z = torch.ones_like(x)  # forward direction
    
    # 방향 벡터 정규화
    direction = torch.stack([X, Y, Z], dim=-1)  # [H, W, 3]
    direction = F.normalize(direction, dim=-1)
    
    # Yaw/Pitch 회전 적용
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    
    # Yaw rotation (Y축 회전)
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    yaw_rot = torch.tensor([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0], 
        [-sin_yaw, 0, cos_yaw]
    ], dtype=direction.dtype)
    
    # Pitch rotation (X축 회전)  
    cos_pitch, sin_pitch = math.cos(pitch_rad), math.sin(pitch_rad)
    pitch_rot = torch.tensor([
        [1, 0, 0],
        [0, cos_pitch, -sin_pitch],
        [0, sin_pitch, cos_pitch]
    ], dtype=direction.dtype)
    
    # 회전 적용
    rotation = yaw_rot @ pitch_rot
    direction_rotated = torch.einsum('hwi,ij->hwj', direction, rotation)
    
    # ERP 구면 좌표로 변환
    x_world, y_world, z_world = direction_rotated[..., 0], direction_rotated[..., 1], direction_rotated[..., 2]
    
    # 구면좌표: theta (경도), phi (위도)
    theta = torch.atan2(x_world, z_world)  # [-π, π]
    phi = torch.asin(torch.clamp(y_world, -1, 1))  # [-π/2, π/2]
    
    # ERP 텍셀 좌표로 변환 (정규화된 [-1, 1] 범위)
    erp_u = theta / math.pi  # [-1, 1]
    erp_v = phi / (math.pi / 2)  # [-1, 1]
    
    # Grid sample 형식으로 스택
    grid = torch.stack([erp_u, erp_v], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    
    return grid

def create_pinhole_to_pinhole_grid(H, W, meta_src, meta_tgt, fovx_deg: float, fovy_deg: float, device=None):
    """
    Create pinhole-to-pinhole warping grid for VICReg-L
    
    Args:
        H, W: Feature map resolution
        meta_src: Source view metadata with yaw/pitch
        meta_tgt: Target view metadata with yaw/pitch  
        fovx_deg, fovy_deg: Field of view angles
        device: Device for tensor creation
        
    Returns:
        grid: [1, H, W, 2] warping grid for F.grid_sample
        vis_mask: [1, 1, H, W] visibility mask
    """
    # Target coordinate grid
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    fx = math.tan(math.radians(fovx_deg) / 2.0)
    fy = math.tan(math.radians(fovy_deg) / 2.0)

    # Target camera rays
    x_t = xs * fx
    y_t = ys * fy
    z_t = torch.ones_like(x_t)
    d_t = torch.stack([x_t, y_t, z_t], dim=-1)
    d_t = F.normalize(d_t, dim=-1)  # (H,W,3)

    def R_yaw_pitch(yaw_deg, pitch_deg):
        yaw = math.radians(yaw_deg); pitch = math.radians(pitch_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        R_y = torch.tensor([[ cy, 0,  sy],
                            [  0, 1,   0],
                            [-sy, 0,  cy]], device=device, dtype=d_t.dtype)
        R_x = torch.tensor([[ 1,  0,  0],
                            [ 0, cp, -sp],
                            [ 0, sp,  cp]], device=device, dtype=d_t.dtype)
        return R_y @ R_x  # yaw -> pitch

    R_src = R_yaw_pitch(meta_src['yaw'], meta_src.get('pitch', 0.0))
    R_tgt = R_yaw_pitch(meta_tgt['yaw'], meta_tgt.get('pitch', 0.0))
    R = R_src.T @ R_tgt                             # R_src^{-1} R_tgt
    d_s = torch.einsum('ij,hwj->hwi', R, d_t)      # target ray -> source camera coordinates

    x_s, y_s, z_s = d_s[...,0], d_s[...,1], d_s[...,2]
    # Project to source image plane (z>0 only valid)
    xs_src = (x_s / (z_s + 1e-8)) / fx             # Normalized [-1,1] range
    ys_src = (y_s / (z_s + 1e-8)) / fy
    grid = torch.stack([xs_src, ys_src], dim=-1).unsqueeze(0)  # [1,H,W,2]

    # Visibility mask (z>0 & |grid|<=1)
    valid = (z_s > 0) & (grid.abs() <= 1).all(dim=-1)
    vis_mask = valid.float().unsqueeze(0).unsqueeze(1)         # [1,1,H,W]
    return grid.clamp(-1,1), vis_mask

def compute_visibility_mask(grid: torch.Tensor, phi_weighting: bool = True) -> torch.Tensor:
    """
    가시성 및 φ-가중 마스크 계산 (ERP 기반 - 레거시 함수)
    
    NOTE: VICReg-L에서는 create_pinhole_to_pinhole_grid가 직접 visibility mask를 반환하므로
    이 함수는 사용되지 않습니다. ERP 기반 워핑에서만 사용됩니다.
    
    Args:
        grid: [1, H, W, 2] warping grid (ERP coordinates)
        phi_weighting: φ 좌표에 cos(φ) 가중치 적용 여부
        
    Returns:
        mask: [1, 1, H, W] 가시성 마스크
    """
    B, H, W, _ = grid.shape
    
    # 기본 가시성: grid 범위 내부
    valid_mask = (grid.abs() <= 1).all(dim=-1)  # [B, H, W]
    
    if phi_weighting:
        # φ-가중: cos(φ) where φ = grid[..., 1] * π/2
        phi = grid[..., 1] * (math.pi / 2)  # [-π/2, π/2]
        cos_phi = torch.cos(phi).abs()
        
        # 중앙 영역에 더 높은 가중치
        mask = valid_mask.float() * cos_phi
    else:
        mask = valid_mask.float()
    
    return mask.unsqueeze(1)  # [B, 1, H, W]

def compute_overlap_consistency(feat_a: torch.Tensor, feat_b: torch.Tensor, 
                               grid_ab: torch.Tensor, vis_mask: torch.Tensor) -> tuple:
    """
    Overlap Consistency Score (OCS) 계산
    
    Args:
        feat_a, feat_b: [B, C, H, W] 특징맵
        grid_ab: [B, H, W, 2] a->b warping grid  
        vis_mask: [B, 1, H, W] 가시성 마스크
        
    Returns:
        ocs: 가중 평균 코사인 유사도
        cos_map: [B, H, W] 코사인 유사도 맵
    """
    # feat_a를 feat_b 좌표계로 warp
    feat_a_warped = F.grid_sample(feat_a, grid_ab, mode='bilinear', 
                                 padding_mode='border', align_corners=False)
    
    # 코사인 유사도 계산
    numerator = (feat_a_warped * feat_b).sum(dim=1)  # [B, H, W]
    denom_a = feat_a_warped.norm(dim=1)  # [B, H, W]
    denom_b = feat_b.norm(dim=1)  # [B, H, W]
    
    cos_map = numerator / (denom_a * denom_b + 1e-8)
    
    # 가시성 마스크 적용
    vis_weight = vis_mask.squeeze(1)  # [B, H, W]
    weighted_cos = cos_map * vis_weight
    
    # 가중 평균
    ocs = weighted_cos.sum() / (vis_weight.sum() + 1e-8)
    
    return ocs, cos_map

# ---------------------------------------------------------------------------
# ‣ VICReg Loss
# ---------------------------------------------------------------------------
class VicRegLoss(nn.Module):
    """파노라마용 VICReg 손실 함수
    
    표준 VICReg 구현:
    - 유사성 손실: MSE 기반
    - 분산 손실: 표준 정규화
    - 공분산 손실: 표준 구현
    
    Args:
        similarity_weight: 유사성 손실 가중치 (기본값: 25.0)
        variance_weight: 분산 손실 가중치 (기본값: 25.0)  
        covariance_weight: 공분산 손실 가중치 (기본값: 1.0)
        use_cosine_sim: 코사인 유사성 사용 여부 (기본값: False)
    """
    def __init__(self, similarity_weight: float = 25.0, variance_weight: float = 25.0, 
                 covariance_weight: float = 1.0, use_cosine_sim: bool = False, use_erm: bool = False):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.use_cosine_sim = use_cosine_sim
        self.use_erm = use_erm  # ERM 적용 여부


    @staticmethod
    def _off_diagonal(x):
        # 공식 구현: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _variance_loss(self, x, eps=1e-4):
        # 표준 분산 손실
        std = torch.sqrt(x.var(dim=0) + eps)
        # 표준편차가 1보다 작으면 페널티
        return torch.mean(F.relu(1.0 - std))

    def _covariance_loss(self, x):
        # 공식 구현: 각 차원간 공분산의 오프다이애고널 제곱 평균
        n, d = x.shape
        if n <= 1:
            return torch.zeros((), device=x.device)
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (n - 1)
        off_diag = self._off_diagonal(cov)
        return (off_diag ** 2).sum() / d

    def forward(self, x, y):
        """표준 VICReg 손실 계산
        
        Args:
            x: 첫 번째 임베딩 (N, D...) - 현재 뷰 오른쪽
            y: 두 번째 임베딩 (N, D...) - 다음 뷰 왼쪽
        Returns:
            total_loss: VICReg 손실
        """
        # 배치 차원으로 평면화 (B, ...) -> (B, D)
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))
        
        return self._compute_standard_vicreg_loss(x, y)
    
    def _compute_erm_vicreg_loss(self, x, y):
        """ERM 기반 VICReg 손실 계산"""
        batch_size = x.shape[0]
        
        # 부드러운 표준화
        x_norm = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-4)
        y_norm = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-4)
        
        # ERM: 샘플별 개별 손실 계산
        individual_losses = []
        
        for i in range(batch_size):
            x_i = x_norm[i:i+1]  # (1, D)
            y_i = y_norm[i:i+1]  # (1, D)
            
            # 1) 샘플별 유사성 손실
            if self.use_cosine_sim:
                cos_sim = F.cosine_similarity(x_i, y_i, dim=1).mean()
                sim_loss_i = 1.0 - cos_sim
            else:
                sim_loss_i = F.mse_loss(x_i, y_i)
            
            # 2) 샘플별 분산 손실 (전체 배치 대비)
            var_loss_i = self._sample_variance_loss(x_i, x_norm) + self._sample_variance_loss(y_i, y_norm)
            
            # 3) 샘플별 공분산 기여도 (근사)
            cov_loss_i = self._sample_covariance_loss(x_i, x_norm) + self._sample_covariance_loss(y_i, y_norm)
            
            # 샘플별 총 손실
            sample_loss = (
                self.similarity_weight * sim_loss_i
                + self.variance_weight * var_loss_i  
                + self.covariance_weight * cov_loss_i
            )
            individual_losses.append(sample_loss)
        
        # ERM: 개별 위험의 경험적 평균
        individual_losses = torch.stack(individual_losses)
        
        # Robust 평균 (이상치 영향 감소)
        # Option 1: 단순 평균
        # total_loss = individual_losses.mean()
        
        # Option 2: Trimmed mean (상위/하위 10% 제거)
        sorted_losses, _ = torch.sort(individual_losses)
        trim_size = max(1, batch_size // 10)
        trimmed_losses = sorted_losses[trim_size:-trim_size] if batch_size > 2*trim_size else sorted_losses
        total_loss = trimmed_losses.mean()
        
        # 안정성 보장
        if not torch.isfinite(total_loss):
            total_loss = x.sum() * 0
            
        return total_loss
    
    def _sample_variance_loss(self, sample, batch):
        """개별 샘플의 분산 기여도 계산 (수정된 안전 버전)"""
        try:
            # 간단화: 표준 분산 손실 사용
            std = torch.sqrt(sample.var(dim=1) + 1e-8)
            return torch.mean(F.relu(0.5 - std))
        except Exception as e:
            print(f"[ERM Debug] Sample variance error: {e}")
            return torch.tensor(0.0, device=sample.device)
    
    def _sample_covariance_loss(self, sample, batch):
        """개별 샘플의 공분산 기여도 계산 (수정된 안전 버전)"""
        try:
            # 간단화: 샘플 크기가 작을 때는 0 리턴
            if sample.shape[1] < 2 or sample.shape[0] < 1:
                return torch.tensor(0.0, device=sample.device)
            return torch.tensor(0.0, device=sample.device)  # 임시로 비활성화
        except Exception as e:
            print(f"[ERM Debug] Sample covariance error: {e}")
            return torch.tensor(0.0, device=sample.device)
    
    def _compute_standard_vicreg_loss(self, x, y):
        """표준 VICReg 손실 (수정됨: var/cov는 정규화 안함)"""
        # (A) sim: 선택적으로 표준화 사용
        if self.use_cosine_sim:
            x_sim = F.normalize(x, dim=1)
            y_sim = F.normalize(y, dim=1)
            sim_loss = 1 - F.cosine_similarity(x_sim, y_sim).mean()
        else:
            x_sim = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-4)
            y_sim = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-4)
            sim_loss = F.mse_loss(x_sim, y_sim)
        
        # (B) var/cov: 평균만 제거 (표준화 금지)
        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)
        
        # 2) 분산 손실
        var_loss = self._variance_loss(x_c) + self._variance_loss(y_c)
        
        # 3) 공분산 손실
        cov_loss = self._covariance_loss(x_c) + self._covariance_loss(y_c)
        
        # 가중 합산
        total_loss = (
            self.similarity_weight * sim_loss
            + self.variance_weight * var_loss
            + self.covariance_weight * cov_loss
        )
        
        # 안정성 보장
        if not torch.isfinite(total_loss):
            total_loss = x.sum() * 0
            
        return total_loss

class VicRegLocalLoss(nn.Module):
    """VICReg-L: 로컬(패치 단위) VICReg 손실 for 동일 광선 대응
    
    핵심 아이디어:
    1. 뷰 메타데이터 기반 geometric alignment (warp)
    2. 겹치는 영역의 대응 토큰들에 대해서만 VICReg 적용
    3. φ-weighting으로 ERP 왜곡 보정
    
    Args:
        inv_weight: Invariance 손실 가중치
        var_weight: Variance 손실 가중치 
        cov_weight: Covariance 손실 가중치
        inv_type: 'l2' 또는 'cos' (INV 손실 유형)
        gamma: 분산 정규화 목표값
    """
    
    def __init__(self, inv_weight: float = 1.0, var_weight: float = 1.0, 
                 cov_weight: float = 0.01, inv_type: str = "l2", gamma: float = 1.0):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight  
        self.cov_weight = cov_weight
        self.inv_type = inv_type
        self.gamma = gamma
        
    def forward(self, features: torch.Tensor, metadata_list: list, 
                batch_size: int, num_views: int) -> dict:
        """
        Args:
            features: [batch*views, C, H, W] 특징맵
            metadata_list: 뷰별 메타데이터 리스트
            batch_size: 배치 크기
            num_views: 뷰 개수
            
        Returns:
            dict: 손실 정보 및 디버그 메트릭
        """
        if num_views <= 1:
            return {
                'loss': torch.zeros((), device=features.device),
                'inv_loss': torch.zeros((), device=features.device),  
                'var_loss': torch.zeros((), device=features.device),
                'cov_loss': torch.zeros((), device=features.device),
                'num_valid_pairs': 0
            }
            
        device = features.device
        B, C, H, W = features.shape
        
        # features를 배치별로 분할: [batch*views, C, H, W] -> [batch, views, C, H, W]
        features_batched = features.view(batch_size, num_views, C, H, W)
        
        total_inv_loss = 0
        total_var_loss = 0  
        total_cov_loss = 0
        num_valid_pairs = 0
        
        # 배치별로 처리
        for b in range(batch_size):
            batch_features = features_batched[b]  # [views, C, H, W]
            batch_metadata = metadata_list[b * num_views:(b + 1) * num_views]
            
            # 인접한 뷰 쌍에 대해 VICReg-L 적용
            for v in range(num_views):
                next_v = (v + 1) % num_views
                
                feat_a = batch_features[v:v+1]      # [1, C, H, W]
                feat_b = batch_features[next_v:next_v+1]  # [1, C, H, W] 
                meta_a = batch_metadata[v]
                meta_b = batch_metadata[next_v]
                
                # Geometric alignment: a -> b 좌표계로 warp
                try:
                    grid_ab, vis_mask = create_pinhole_to_pinhole_grid(
                        H, W, meta_src=meta_a, meta_tgt=meta_b,
                        fovx_deg=meta_b.get('fovx', meta_b['effective_fov']),
                        fovy_deg=meta_b.get('fovy', meta_b['effective_fov']),
                        device=device
                    )
                    
                    # feat_a를 meta_b 좌표계로 정렬
                    feat_a_warped = F.grid_sample(feat_a, grid_ab, mode='bilinear',
                                                 padding_mode='border', align_corners=False)
                    
                    # 유효한 겹침 영역만 추출
                    mask = vis_mask.squeeze()  # [H, W]
                    if mask.sum() < 10:  # 최소 패치 개수
                        continue
                        
                    # 로컬 VICReg 손실 계산
                    inv, var, cov = self._compute_vicreg_local(
                        feat_a_warped, feat_b, vis_mask
                    )
                    
                    total_inv_loss += inv
                    total_var_loss += var
                    total_cov_loss += cov
                    num_valid_pairs += 1
                    
                except Exception as e:
                    print(f"VICReg-L warning: {e}")
                    continue
        
        # 평균화
        if num_valid_pairs > 0:
            total_inv_loss /= num_valid_pairs
            total_var_loss /= num_valid_pairs  
            total_cov_loss /= num_valid_pairs
        
        # 가중 합산
        total_loss = (
            self.inv_weight * total_inv_loss +
            self.var_weight * total_var_loss +
            self.cov_weight * total_cov_loss
        )
        
        return {
            'loss': total_loss,
            'inv_loss': total_inv_loss,
            'var_loss': total_var_loss, 
            'cov_loss': total_cov_loss,
            'num_valid_pairs': num_valid_pairs
        }
    
    def _compute_vicreg_local(self, z_a: torch.Tensor, z_b: torch.Tensor, 
                             mask: torch.Tensor) -> tuple:
        """
        겹침 대응 토큰에 대한 로컬 VICReg 손실
        
        Args:
            z_a, z_b: [1, C, H, W] (a는 warp 후 b 좌표계에 정렬됨)
            mask: [1, 1, H, W] 겹침/가시성 마스크
            
        Returns:
            (inv_loss, var_loss, cov_loss)
        """
        B, C, H, W = z_a.shape
        
        # 마스크가 적용된 패치들만 추출
        m = mask.squeeze() > 0  # [H, W]
        if m.sum() == 0:
            device = z_a.device
            return (
                torch.zeros((), device=device),
                torch.zeros((), device=device), 
                torch.zeros((), device=device)
            )
            
        # [B, C, H, W] -> [B, H, W, C] -> [N, C] (N = valid patches)
        Za = z_a.permute(0, 2, 3, 1)[0][m]  # [N, C]
        Zb = z_b.permute(0, 2, 3, 1)[0][m]  # [N, C]
        
        # 1. INV (Invariance): 대응 토큰 간 거리
        if self.inv_type == "cos":
            # 코사인 거리 (1 - cosine_similarity)
            cos_sim = (Za * Zb).sum(-1) / (Za.norm(dim=-1) * Zb.norm(dim=-1) + 1e-8)
            L_inv = (1 - cos_sim).mean()
        else:
            # L2 거리
            L_inv = ((Za - Zb) ** 2).mean()
        
        # 2. VAR (Variance): 각 분기에서 특징 차원별 표준편차 >= gamma
        def _var_loss(Z):
            if Z.shape[0] <= 1:
                return torch.zeros((), device=Z.device)
            std = Z.float().std(dim=0)  # [C]
            return F.relu(self.gamma - std).mean()
            
        L_var = _var_loss(Za) + _var_loss(Zb)
        
        # 3. COV (Covariance): 오프대각 공분산을 0에 가깝게
        def _cov_loss(Z):
            if Z.shape[0] <= 1 or Z.shape[1] <= 1:
                return torch.zeros((), device=Z.device)
            Z_centered = Z - Z.mean(dim=0, keepdim=True)
            N = Z.shape[0]
            COV = (Z_centered.T @ Z_centered) / (N - 1)  # [C, C]
            # 오프대각 원소들의 제곱 평균
            off_diag = COV - torch.diag(torch.diag(COV))
            return (off_diag ** 2).sum() / (COV.shape[0] ** 2)
            
        L_cov = _cov_loss(Za) + _cov_loss(Zb)
        
        return L_inv, L_var, L_cov

# ---------------------------------------------------------------------------
# ‣ PanoramaVLM (VICReg + AR)
# ---------------------------------------------------------------------------
class PanoramaVLM(nn.Module):
    """파노라마 비전-언어 모델
    
    3단계 학습 구조:
    1) Vision: VICReg을 통한 파노라마 뷰 간 중첩 영역 일관성 학습
    2) Resampler: 리샘플러 + 투영 레이어 학습 (AR loss)
    3) Finetune: 전체 모델 파인튜닝 (AR loss)
    
    개선된 텍스트 처리:
    - 안정적인 토크나이저 설정
    - 올바른 어텐션 마스크 처리
    - 개선된 레이블 시프트
    - 강화된 생성 함수
    """
    
    def __init__(
        self,
        vision_name: str = "google/siglip-base-patch16-224",
        language_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        resampler_type: str = "mlp",
        latent_dimension: int = 768,
        vicreg_loss_weight: float = 1.0,
        vicreg_overlap_ratio: float = 0.5,
        max_text_length: int = 512,
        # VICReg Local 파라미터들
        use_vicreg_local: bool = False,
        vicreg_local_weight: float = 0.5,
        vicreg_local_inv_weight: float = 1.0,
        vicreg_local_var_weight: float = 1.0,
        vicreg_local_cov_weight: float = 0.01,
        vicreg_local_inv_type: str = "l2",
        vicreg_local_gamma: float = 1.0,
        config: Optional['ModelConfig'] = None,
    ):
        super().__init__()

        # 설정 시스템 통합 ------------------------------------------------
        if config is not None:
            # config가 제공된 경우 config 우선 사용
            vision_name = config.vision_name
            language_model_name = config.language_model_name
            resampler_type = config.resampler_type
            latent_dimension = config.latent_dimension
            vicreg_loss_weight = config.vicreg_loss_weight
            vicreg_overlap_ratio = config.vicreg_overlap_ratio
            max_text_length = config.max_text_length
            # VICReg Local 파라미터들
            use_vicreg_local = config.use_vicreg_local
            vicreg_local_weight = config.vicreg_local_weight
            vicreg_local_inv_weight = config.vicreg_local_inv_weight
            vicreg_local_var_weight = config.vicreg_local_var_weight
            vicreg_local_cov_weight = config.vicreg_local_cov_weight
            vicreg_local_inv_type = config.vicreg_local_inv_type
            vicreg_local_gamma = config.vicreg_local_gamma
            self.config = config
            print(f"[Model] Using provided ModelConfig: {config}")
        else:
            # config가 없는 경우 개별 파라미터로 config 생성
            from .config import ModelConfig
            self.config = ModelConfig(
                vision_name=vision_name,
                language_model_name=language_model_name,
                resampler_type=resampler_type,
                latent_dimension=latent_dimension,
                vicreg_loss_weight=vicreg_loss_weight,
                vicreg_overlap_ratio=vicreg_overlap_ratio,
                max_text_length=max_text_length,
                # VICReg Local 파라미터들
                use_vicreg_local=use_vicreg_local,
                vicreg_local_weight=vicreg_local_weight,
                vicreg_local_inv_weight=vicreg_local_inv_weight,
                vicreg_local_var_weight=vicreg_local_var_weight,
                vicreg_local_cov_weight=vicreg_local_cov_weight,
                vicreg_local_inv_type=vicreg_local_inv_type,
                vicreg_local_gamma=vicreg_local_gamma,
            )
            print(f"[Model] Created ModelConfig from parameters")

        # 비전 인코더 초기화 ------------------------------------------------
        self.vision_encoder = AutoModel.from_pretrained(vision_name, trust_remote_code=True)
        if hasattr(self.vision_encoder, "vision_model"):
            self.vision_encoder = self.vision_encoder.vision_model
        vision_hidden_size = self._get_vision_hidden_size(self.vision_encoder)
        # VICReg 정규화 레이어 (per-call BatchNorm 제거 → LayerNorm 사용)
        self.vicreg_norm = nn.LayerNorm(vision_hidden_size)
        
        # 리샘플러 초기화 ---------------------------------------------------
        if resampler_type == "mlp":
            self.resampler = MLPResampler(vision_hidden_size, latent_dimension)
        else:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {resampler_type}")

        # 언어 모델 초기화 ---------------------------------------------------
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        
        # 비전 특징을 언어 모델 임베딩 공간으로 투영하는 레이어
        self.vision_to_language_projection = nn.Linear(
            latent_dimension, 
            self.language_model.config.hidden_size
        )

        # 토크나이저 초기화 및 설정 개선 ------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self._setup_tokenizer()
        
        # 특수 토큰 정의
        self.vision_token_id = self.tokenizer.convert_tokens_to_ids("<|vision|>")
        if self.vision_token_id == self.tokenizer.unk_token_id:
            self.vision_token_id = self.tokenizer.bos_token_id
        
        # UniversalTextFormatter 초기화 (생성 시 사용)
        try:
            from .processors.universal_text_formatter import UniversalTextFormatter
            self._text_formatter = UniversalTextFormatter(
                tokenizer_name_or_path=language_model_name,
                system_msg="You are an expert assistant specialized in analyzing panoramic images."
            )
            print(f"[Model] Initialized UniversalTextFormatter for {self._text_formatter.model_family}")
        except Exception as e:
            print(f"[Model] Failed to initialize UniversalTextFormatter: {e}")
            self._text_formatter = None

        # VICReg 손실 함수 및 가중치 / 중첩 비율 ---------------------------
        self.vicreg_loss = VicRegLoss(
            similarity_weight=15.0,
            variance_weight=15.0,
            covariance_weight=1.0,
            use_cosine_sim=False,
            use_erm=False
        )
        
        # VICReg-L (Local VICReg) 손실 함수 추가
        if config:
            self.vicreg_local_loss = VicRegLocalLoss(
                inv_weight=config.vicreg_local_inv_weight,
                var_weight=config.vicreg_local_var_weight,
                cov_weight=config.vicreg_local_cov_weight,
                inv_type=config.vicreg_local_inv_type,
                gamma=config.vicreg_local_gamma
            )
            self.use_vicreg_local = config.use_vicreg_local
            self.vicreg_local_weight = config.vicreg_local_weight
        else:
            self.vicreg_local_loss = VicRegLocalLoss(
                inv_weight=1.0,
                var_weight=1.0, 
                cov_weight=0.01,
                inv_type="l2",
                gamma=1.0
            )
            self.use_vicreg_local = False
            self.vicreg_local_weight = 0.5
        
        self.vicreg_loss_weight = vicreg_loss_weight
        self.vicreg_overlap_ratio = vicreg_overlap_ratio
        
        # 텍스트 처리 관련 설정
        self.max_text_length = max_text_length
        self.ignore_index = -100
        # num_views 경고 플래그
        self._warned_single_view = False
        # Loss 검증 디버깅 플래그 (개발/디버깅 시에만 활성화)
        self._debug_loss_verification = False
        

    def _setup_tokenizer(self):
        """토크나이저 설정 강화 - 모든 특수 토큰 안전 설정"""
        original_vocab_size = len(self.tokenizer)
        tokens_added = False
        
        # 1. 패딩 토큰 설정 (pad = eos 원칙)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[Tokenizer Setup] Set pad_token = eos_token: '{self.tokenizer.eos_token}'")
            else:
                # EOS도 없는 경우 둘 다 추가
                new_tokens = {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}
                self.tokenizer.add_special_tokens(new_tokens)
                tokens_added = True
                print(f"[Tokenizer Setup] Added eos_token and pad_token: '<|endoftext|>'")
        
        # 2. BOS 토큰 확인
        if self.tokenizer.bos_token is None:
            print(f"[Tokenizer Setup] Warning: No bos_token defined")
        
        # 3. UNK 토큰 확인
        if self.tokenizer.unk_token is None:
            print(f"[Tokenizer Setup] Warning: No unk_token defined")
        
        # 4. 필수 특수 토큰 추가
        special_tokens_to_add = []
        vision_token = '<|vision|>'
        
        # 비전 토큰이 없으면 추가
        if not any(vision_token in str(token) for token in self.tokenizer.additional_special_tokens):
            special_tokens_to_add.append(vision_token)
        
        # endoftext 토큰 확인 (EOS와 별도)
        if self.tokenizer.eos_token != '<|endoftext|>':
            if not any('<|endoftext|>' in str(token) for token in self.tokenizer.additional_special_tokens):
                special_tokens_to_add.append('<|endoftext|>')
        
        # 필요한 토큰들 추가
        if special_tokens_to_add:
            added_tokens = self.tokenizer.add_special_tokens({
                'additional_special_tokens': special_tokens_to_add
            })
            if added_tokens > 0:
                tokens_added = True
                print(f"[Tokenizer Setup] Added {added_tokens} special tokens: {special_tokens_to_add}")
        
        # 5. 언어 모델 임베딩 크기 조정 (토큰이 추가된 경우)
        if tokens_added:
            old_embeddings = self.language_model.get_input_embeddings().weight.size(0)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            new_embeddings = self.language_model.get_input_embeddings().weight.size(0)
            print(f"[Tokenizer Setup] Resized embeddings: {old_embeddings} -> {new_embeddings}")
        
        # 6. 패딩 방향 설정 (학습 시는 right)
        self.tokenizer.padding_side = "right"
        
        # 8. 중요한 토큰 ID들 저장 (빠른 접근용)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        
        # 패딩과 EOS가 동일한지 확인
        if self.pad_token_id == self.eos_token_id:
            print(f"[Tokenizer Setup] ✓ pad_token_id == eos_token_id (consistent with loss masking)")

    # ---------------- 유틸리티 함수 ---------------------------------------------
    @staticmethod
    def _get_vision_hidden_size(vision_model: nn.Module) -> int:
        """비전 모델의 은닉 차원 크기를 추출"""
        possible_keys = ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]
        for key in possible_keys:
            if hasattr(vision_model.config, key):
                return getattr(vision_model.config, key)
        raise AttributeError("비전 모델의 은닉 차원 크기를 찾을 수 없습니다")
    
    def _has_cls_token(self, vision_output: torch.Tensor) -> bool:
        """비전 인코더 출력에 CLS 토큰(첫 토큰)이 포함되어 있는지 휴리스틱 판별.
        1) 비전 인코더 모듈에 cls_token/class_token/class_embedding 속성이 있으면 True
        2) 시퀀스 길이 L 또는 L-1 이 완전제곱인지 검사 (패치 그리드 여부)
           - (L 가 완전제곱) -> CLS 없다고 가정
           - (L-1 이 완전제곱) -> CLS 있다고 가정
        3) 그 외에는 False (보수적으로 전체를 패치로 간주)
        """
        for attr in ("cls_token", "class_token", "class_embedding"):
            if hasattr(self.vision_encoder, attr):
                return True
        seq_len = vision_output.size(1)
        if seq_len > 1:
            if int(math.isqrt(seq_len)) ** 2 == seq_len:
                return False
            if int(math.isqrt(seq_len - 1)) ** 2 == (seq_len - 1):
                return True
        return False
    
    def _prepare_text_inputs(self, input_ids, attention_mask, labels=None):
        """텍스트 입력 전처리 및 검증"""
        batch_size = input_ids.size(0)
        
        # 어텐션 마스크가 없으면 생성
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 레이블이 없으면 input_ids를 복사해서 생성 (다음 토큰 예측용)
        if labels is None:
            labels = input_ids.clone()
        
        # 입력 길이 제한
        max_len = min(input_ids.size(1), self.max_text_length)
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        labels = labels[:, :max_len]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'batch_size': batch_size
        }
    
    def _create_vision_attention_mask(self, batch_size, vision_seq_len, device):
        """비전 토큰용 어텐션 마스크 생성"""
        return torch.ones(batch_size, vision_seq_len, dtype=torch.long, device=device)
    
    def _create_combined_inputs_for_training(self, vision_tokens, text_inputs):
        """학습용 비전 토큰과 텍스트 입력을 결합"""
        batch_size = text_inputs['batch_size']
        device = vision_tokens.device
        
        # 텍스트 임베딩 생성
        text_embeddings = self.language_model.get_input_embeddings()(text_inputs['input_ids'])
        
        # 배치 크기 일치 확인
        if vision_tokens.size(0) != batch_size:
            min_batch = min(vision_tokens.size(0), batch_size)
            vision_tokens = vision_tokens[:min_batch]
            text_embeddings = text_embeddings[:min_batch]
            for key in text_inputs:
                if torch.is_tensor(text_inputs[key]) and text_inputs[key].size(0) > min_batch:
                    text_inputs[key] = text_inputs[key][:min_batch]
            batch_size = min_batch
        
        # 임베딩 결합
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
        
        # 어텐션 마스크 결합
        vision_attention = self._create_vision_attention_mask(
            batch_size, vision_tokens.size(1), device
        )
        combined_attention = torch.cat([vision_attention, text_inputs['attention_mask']], dim=1)
        
        # 레이블 결합 개선
        vision_labels = torch.full(
            (batch_size, vision_tokens.size(1)), 
            self.ignore_index, 
            dtype=text_inputs['labels'].dtype, 
            device=device
        )
        
        # 텍스트 레이블은 시프트하지 않고 그대로 사용
        # (다음 토큰 예측은 언어모델 내부에서 자동으로 처리됨)
        combined_labels = torch.cat([vision_labels, text_inputs['labels']], dim=1)
        
        # 디버깅 정보 출력
        # valid_text_labels = (text_inputs['labels'] != self.ignore_index).sum()
        # print(f"[AR Debug] Valid text labels: {valid_text_labels.item()}")
        # print(f"[AR Debug] Vision labels shape: {vision_labels.shape}")
        # print(f"[AR Debug] Text labels shape: {text_inputs['labels'].shape}")
        
        return {
            'inputs_embeds': combined_embeddings,
            'attention_mask': combined_attention,
            'labels': combined_labels
        }

    # ==================== 메인 순전파 및 생성 함수 ====================
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "vision",
        view_metadata: Optional[list] = None,  # VICReg-L을 위한 메타데이터
        **kwargs
    ):
        """
        파노라마 VLM 순전파 - 3단계 학습 지원
        
        Args:
            pixel_values: 파노라마 이미지 픽셀 값 (B, V, C, H, W) 또는 (B, C, H, W)
            input_ids: 텍스트 토큰 ID (stage가 resampler/finetune일 때 필수)
            attention_mask: 텍스트 어텐션 마스크
            labels: 텍스트 레이블 (stage가 resampler/finetune일 때 필수)
            stage: 학습 단계 - "vision" (VICReg), "resampler", "finetune"
            
        Returns:
            dict: 손실 및 관련 메트릭
        """
        # 통합된 전처리 사용
        batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
        vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)
        
        if stage == "vision":
            if num_views <= 1 and not self._warned_single_view:
                print("[VICReg] Warning: num_views <= 1, VICReg 손실이 0이 됩니다.")
                self._warned_single_view = True
            
            overlap_ratio = kwargs.get('overlap_ratio', self.vicreg_overlap_ratio)
            
            # 기존 VICReg 손실
            vicreg_raw = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, overlap_ratio=overlap_ratio
            )
            vicreg_loss = vicreg_raw * self.vicreg_loss_weight
            
            total_loss = vicreg_loss
            loss_dict = {
                "loss": total_loss,
                "vicreg_loss": vicreg_loss,
                "vicreg_raw": vicreg_raw.detach(),
                "vicreg_weight": self.vicreg_loss_weight
            }
            
            # VICReg-L 손실 추가 (메타데이터가 있고 활성화된 경우)
            if self.use_vicreg_local and view_metadata is not None:
                try:
                    # 특징맵을 spatial format으로 변환 (필요시)
                    # vision_hidden_states: [B*V, seq_len, hidden_dim]
                    # -> [B*V, C, H, W] for VICReg-L
                    
                    # 패치 토큰에서 공간 특징맵으로 변환
                    feat_spatial = self._convert_to_spatial_features(vision_hidden_states, batch_size, num_views)
                    
                    vicreg_local_result = self.vicreg_local_loss(
                        feat_spatial, view_metadata, batch_size, num_views
                    )
                    
                    vicreg_local_loss = vicreg_local_result['loss'] * self.vicreg_local_weight
                    total_loss = total_loss + vicreg_local_loss
                    
                    # 로깅 정보 추가
                    loss_dict.update({
                        "vicreg_local_loss": vicreg_local_loss,
                        "vicreg_local_inv": vicreg_local_result['inv_loss'].detach(),
                        "vicreg_local_var": vicreg_local_result['var_loss'].detach(),
                        "vicreg_local_cov": vicreg_local_result['cov_loss'].detach(),
                        "vicreg_local_pairs": vicreg_local_result['num_valid_pairs'],
                        "vicreg_local_weight": self.vicreg_local_weight
                    })
                    
                    print(f"[VICReg-L] Added local loss: {vicreg_local_loss.item():.6f} "
                          f"(pairs: {vicreg_local_result['num_valid_pairs']})")
                    
                except Exception as e:
                    print(f"[VICReg-L] Warning: Failed to compute local loss: {e}")
            
            loss_dict["loss"] = total_loss
            return loss_dict
        
        # 리샘플러 통과
        resampled_features = self.resampler(vision_hidden_states)  # [B*V, S, D]
        
        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            
            # Fix: 멀티뷰 토큰을 시퀀스로 펼쳐서 모든 뷰 보존
            S, D = resampled_features.size(1), resampled_features.size(2)
            resampled_features = resampled_features.view(batch_size, num_views * S, D)  # [B, V*S, D]
            
            return self._compute_autoregressive_loss(
                resampled_features, input_ids, attention_mask, labels
            )
        else:
            raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    # ==================== 비전 특징 처리 헬퍼 함수들 ====================
    def _convert_to_spatial_features(self, vision_hidden_states: torch.Tensor, 
                                   batch_size: int, num_views: int) -> torch.Tensor:
        """
        패치 토큰을 공간 특징맵으로 변환
        
        Args:
            vision_hidden_states: [B*V, seq_len, hidden_dim] (seq_len includes CLS token)
            batch_size: 배치 크기
            num_views: 뷰 개수
            
        Returns:
            spatial_features: [B*V, C, H, W] 공간 특징맵
        """
        BV, seq_len, C = vision_hidden_states.shape
        
        # CLS 토큰 제거 (있는 경우만)
        has_cls = self._has_cls_token(vision_hidden_states)
        patch_tokens = vision_hidden_states[:, 1:, :] if has_cls else vision_hidden_states
        num_patches = patch_tokens.shape[1]
        
        # 패치 수로부터 H, W 계산 (정사각형 가정)
        H = W = int(math.sqrt(num_patches))
        if H * W != num_patches:
            # 정사각형이 아닌 경우 _infer_hw 사용
            H, W = _infer_hw(num_patches)
        
        # [B*V, num_patches, C] -> [B*V, C, H, W]
        spatial_features = patch_tokens.transpose(1, 2).reshape(BV, C, H, W)
        
        return spatial_features

    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,   # (배치*뷰수, 패치수, 차원)
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,
    ):
        if num_views <= 1:
            return torch.zeros((), device=vision_output.device)
            
        # 디버그 정보 (한 번만 출력)
        if not hasattr(self, '_debug_printed'):
            print(f"[VICReg Debug] vision_output shape: {vision_output.shape}")
            print(f"[VICReg Debug] batch_size: {batch_size}, num_views: {num_views}")
            self._debug_printed = True
        
        # CLS 토큰 처리
        has_cls_token = self._has_cls_token(vision_output)
        patch_features = vision_output[:, 1:] if has_cls_token else vision_output
        num_patches = patch_features.size(1)
        
        if not hasattr(self, '_debug_printed2'):
            print(f"[VICReg Debug] has_cls_token: {has_cls_token}, num_patches: {num_patches}")
        
        grid_height, grid_width = _infer_hw(num_patches)
        grid_features = patch_features.view(batch_size, num_views, grid_height, grid_width, -1)
        overlap_columns = max(1, int(grid_width * overlap_ratio))
        
        if not hasattr(self, '_debug_printed2'):
            print(f"[VICReg Debug] grid: {grid_height}x{grid_width}, overlap_columns: {overlap_columns}")
            self._debug_printed2 = True
        
        right = grid_features[..., -overlap_columns:, :]  # 현재 뷰의 오른쪽
        left = torch.roll(grid_features, shifts=-1, dims=1)[..., :overlap_columns, :]  # 다음 뷰의 왼쪽
        right_flat = right.reshape(-1, right.shape[-1])
        left_flat = left.reshape(-1, left.shape[-1])
        
        if not hasattr(self, '_debug_printed3'):
            print(f"[VICReg Debug] right_flat shape: {right_flat.shape}, left_flat shape: {left_flat.shape}")
            print(f"[VICReg Debug] right_flat mean: {right_flat.mean().item():.6f}, std: {right_flat.std().item():.6f}")
            print(f"[VICReg Debug] left_flat mean: {left_flat.mean().item():.6f}, std: {left_flat.std().item():.6f}")
            print(f"[VICReg Debug] diff abs mean: {(right_flat - left_flat).abs().mean().item():.6f}")
            self._debug_printed3 = True
        
        if right_flat.shape[0] == 0:
            return torch.zeros((), device=vision_output.device)
        
        vicreg_loss = self.vicreg_loss(right_flat, left_flat)
        
        if not hasattr(self, '_debug_printed4'):
            print(f"[VICReg Debug] Final VICReg loss: {vicreg_loss.item():.6f}")
            self._debug_printed4 = True
            
        return vicreg_loss

    # ==================== 자기회귀 손실 계산 함수 ====================
    def _compute_autoregressive_loss(self, image_features, input_ids, attention_mask, labels):
        # 1. 비전 특징을 언어 모델 임베딩 공간으로 투영
        vision_tokens = self._project_vision_tokens(image_features)
        # 2. 텍스트 입력 전처리
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels)
        # 3. 비전과 텍스트 입력 결합
        combined_inputs = self._create_combined_inputs_for_training(vision_tokens, text_inputs)
        # 4. 언어 모델 순전파
        try:
            outputs = self.language_model(
                inputs_embeds=combined_inputs['inputs_embeds'],
                attention_mask=combined_inputs['attention_mask'],
                labels=combined_inputs['labels'],
                return_dict=True
            )
            if not torch.isfinite(outputs.loss):
                manual = self._compute_manual_cross_entropy_loss(
                    outputs.logits, combined_inputs['labels']
                )
                manual.setdefault('ar_loss', manual['loss'])
                return manual
            
            # Loss 검증: 수동 계산과 비교 (디버깅용)
            if hasattr(self, '_debug_loss_verification') and self._debug_loss_verification:
                manual = self._compute_manual_cross_entropy_loss(
                    outputs.logits, combined_inputs['labels']
                )
                diff = abs(outputs.loss.item() - manual['loss'].item())
                if diff > 0.01:  # 차이가 큰 경우 경고
                    print(f"[Loss Debug] HF loss: {outputs.loss.item():.6f}, Manual loss: {manual['loss'].item():.6f}, Diff: {diff:.6f}")
            
            return {"loss": outputs.loss, "ar_loss": outputs.loss, "logits": outputs.logits}
        except Exception:
            fallback_loss = torch.tensor(0.0, device=vision_tokens.device, requires_grad=True)
            return {
                "loss": fallback_loss,
                "ar_loss": fallback_loss,
                "logits": torch.zeros(
                    (text_inputs['batch_size'], combined_inputs['inputs_embeds'].size(1), self.language_model.config.vocab_size),
                    device=vision_tokens.device
                )
            }

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 32, temperature: float = 0.7, **kwargs):
        """
        간소화된 파노라마 이미지 텍스트 생성 함수 - UniversalTextFormatter 사용
        
        Args:
            pixel_values: 파노라마 이미지 픽셀 값
            input_ids: 입력 텍스트 토큰 ID (데이터셋에서 제공되어야 함)
            attention_mask: 입력 텍스트용 어텐션 마스크
            max_new_tokens: 생성할 최대 새 토큰 수
            temperature: 샘플링 온도
        
        Returns:
            dict: {"generated_ids", "text"}
        """
        self.eval()
        
        try:
            # 1. 입력 전처리 (forward와 동일한 방식)
            batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
            vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)
            vision_tokens = self._process_vision_features_for_generation(vision_hidden_states, batch_size, num_views)
            
            # 2. 텍스트 입력 준비
            input_ids, attention_mask = self._prepare_generation_inputs(
                input_ids, attention_mask, batch_size, normalized_pixels.device
            )
            
            # 3. 생성 실행
            return self._execute_generation(
                vision_tokens, input_ids, attention_mask, 
                max_new_tokens, temperature, **kwargs
            )
            
        except Exception as e:
            print(f"[Generate] Error in generation: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_generation_result(batch_size, pixel_values.device)
    
    # ==================== 비전 및 텍스트 전처리 헬퍼 함수들 ====================
    def _normalize_pixel_values(self, pixel_values):
        """픽셀 값 차원 정규화 - forward와 generate 공통 로직"""
        if pixel_values.ndim == 5:
            batch_size, num_views, _, _, _ = pixel_values.shape
        elif pixel_values.ndim == 4:
            batch_size, _, _, _ = pixel_values.shape
            num_views = 1
            pixel_values = pixel_values.unsqueeze(1)  # 뷰 차원 추가
        else:
            raise ValueError(f"pixel_values shape invalid: {pixel_values.shape}")
        
        return batch_size, num_views, pixel_values
    
    def _extract_vision_features(self, pixel_values, batch_size, num_views):
        """통합된 비전 특징 추출 - forward와 generate 공통 파이프라인"""
        # 1. 픽셀값을 비전 인코더 입력 형태로 변환
        flattened_pixel_values = pixel_values.view(batch_size * num_views, *pixel_values.shape[2:])
        
        # 2. 비전 인코더 통과
        vision_output = self.vision_encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state  # (B*V, P, D)
        
        # 3. VICReg 정규화
        vision_hidden_states = self.vicreg_norm(vision_hidden_states)
        
        return vision_hidden_states
    
    def _process_vision_features_for_generation(self, vision_hidden_states, batch_size, num_views):
        """생성용 비전 특징 후처리 - 리샘플러 + 투영"""
        # 1. 리샘플러 통과
        resampled_features = self.resampler(vision_hidden_states)
        
        # 2. 배치 차원 복원 (멀티뷰 처리)
        seq_len = resampled_features.size(1)
        feature_dim = resampled_features.size(2)
        resampled_features = resampled_features.view(batch_size, num_views * seq_len, feature_dim)
        
        # 3. 언어 모델 임베딩 공간으로 투영
        vision_tokens = self._project_vision_tokens(resampled_features)
        
        return vision_tokens
    
    # ==================== 생성용 입력 준비 헬퍼 함수들 ====================
    def _prepare_generation_inputs(self, input_ids, attention_mask, batch_size, device):
        """생성을 위한 텍스트 입력 준비"""
        if input_ids is None:
            print("[Generate] Warning: input_ids not provided, creating default prompt")
            input_ids, attention_mask = self._create_default_prompt(batch_size, device)
        else:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                attention_mask = (input_ids != pad_id).long()
            
            # 배치 크기 맞춤
            input_ids, attention_mask = self._adjust_batch_size(
                input_ids, attention_mask, batch_size
            )
        
        return input_ids, attention_mask
    
    def _create_default_prompt(self, batch_size, device):
        """기본 프롬프트 생성"""
        DEFAULT_PROMPT = "Describe this panoramic image in detail."
        DEFAULT_MAX_LENGTH = 64
        
        encoding = self.tokenizer(
            DEFAULT_PROMPT,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        return self._adjust_batch_size(input_ids, attention_mask, batch_size)
    
    def _adjust_batch_size(self, input_ids, attention_mask, target_batch_size):
        """입력을 목표 배치 크기에 맞춤"""
        if input_ids.size(0) == 1 and target_batch_size > 1:
            input_ids = input_ids.expand(target_batch_size, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.expand(target_batch_size, -1)
        return input_ids, attention_mask
    
    def _execute_generation(self, vision_tokens, input_ids, attention_mask, 
                           max_new_tokens, temperature, **kwargs):
        """실제 생성 실행 - UniversalTextFormatter와 연동"""
        # 1. 입력 준비
        combined_inputs = self._create_combined_inputs_for_generation(vision_tokens, input_ids, attention_mask)
        
        # 2. 생성 파라미터 설정
        generation_kwargs = self._build_generation_kwargs(
            combined_inputs, max_new_tokens, temperature, **kwargs
        )
        
        # 3. 생성 실행
        generated_ids = self.language_model.generate(**generation_kwargs)
        
        # 4. 텍스트 후처리
        return self._postprocess_generated_text(generated_ids)
    
    def _get_tokenizer_info(self):
        """LoRA 호환 토크나이저 정보 추출"""
        try:
            # 언어 모델에서 토크나이저 정보 추출
            if hasattr(self.language_model, 'config'):
                config = self.language_model.config
            elif hasattr(self.language_model, 'base_model') and hasattr(self.language_model.base_model, 'config'):
                config = self.language_model.base_model.config
            else:
                config = None
            
            if config:
                pad_token_id = getattr(config, 'pad_token_id', None)
                eos_token_id = getattr(config, 'eos_token_id', None)
                bos_token_id = getattr(config, 'bos_token_id', None)
                
                return {
                    'pad_token_id': pad_token_id,
                    'eos_token_id': eos_token_id, 
                    'bos_token_id': bos_token_id
                }
            
            return {}
        except Exception:
            return {}
    
    def _create_combined_inputs_for_generation(self, vision_tokens, input_ids, attention_mask):
        """생성용 비전과 텍스트 입력 결합"""
        device = vision_tokens.device
        
        # 텍스트 임베딩 생성 (LoRA 호환)
        if hasattr(self.language_model, 'get_input_embeddings'):
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        elif hasattr(self.language_model, 'base_model'):
            # LoRA가 적용된 경우 base_model을 통해 접근
            text_embeddings = self.language_model.base_model.get_input_embeddings()(input_ids)
        else:
            # 폴백: 직접 임베딩 레이어 접근
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        # 배치 크기 정렬
        batch_size = min(vision_tokens.size(0), text_embeddings.size(0))
        vision_tokens = vision_tokens[:batch_size]
        text_embeddings = text_embeddings[:batch_size]
        attention_mask = attention_mask[:batch_size]
        
        # 임베딩 결합
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
        
        # 어텐션 마스크 결합
        vision_mask = torch.ones(
            batch_size, vision_tokens.size(1), 
            dtype=torch.long, device=device
        )
        combined_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        return {
            'inputs_embeds': combined_embeddings,
            'attention_mask': combined_attention_mask
        }
    
    def _build_generation_kwargs(self, combined_inputs, max_new_tokens, temperature, **kwargs):
        """생성 파라미터 구성"""
        # 온도 정규화
        temperature = max(0.1, min(temperature, 1.0))
        
        # 기본 생성 설정
        MIN_NEW_TOKENS = 5
        DEFAULT_TOP_P = 0.9
        DEFAULT_TOP_K = 50
        DEFAULT_REP_PENALTY = 1.1
        DEFAULT_LENGTH_PENALTY = 1.0
        
        # LoRA 호환 토크나이저 설정
        tokenizer_info = self._get_tokenizer_info()
        
        generation_kwargs = {
            **combined_inputs,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': kwargs.get('min_new_tokens', MIN_NEW_TOKENS),
            'temperature': temperature,
            'do_sample': temperature > 0.1,
            'top_p': kwargs.get('top_p', DEFAULT_TOP_P),
            'top_k': kwargs.get('top_k', DEFAULT_TOP_K),
            'repetition_penalty': kwargs.get('repetition_penalty', DEFAULT_REP_PENALTY),
            'length_penalty': kwargs.get('length_penalty', DEFAULT_LENGTH_PENALTY),
        }
        
        # 토크나이저 정보 추가 (LoRA 호환)
        if tokenizer_info.get('pad_token_id') is not None:
            generation_kwargs['pad_token_id'] = tokenizer_info['pad_token_id']
        if tokenizer_info.get('eos_token_id') is not None:
            generation_kwargs['eos_token_id'] = tokenizer_info['eos_token_id']
        
        # 사용자 지정 토크나이저 설정 오버라이드
        for key in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
            if key in kwargs:
                generation_kwargs[key] = kwargs[key]
        
        return generation_kwargs
    
    def _postprocess_generated_text(self, generated_ids):
        """생성된 텍스트 후처리"""
        # 기본 디코딩
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        # UniversalTextFormatter 적용 (가능한 경우)
        if hasattr(self, '_text_formatter'):
            cleaned_texts = [
                self._text_formatter.extract_assistant_response(text) 
                for text in generated_texts
            ]
        else:
            cleaned_texts = [self._simple_clean_text(text) for text in generated_texts]
        
        return {
            "generated_ids": generated_ids,
            "text": cleaned_texts
        }
    
    def _simple_clean_text(self, text):
        """기본 텍스트 정리 (fallback)"""
        return text.strip()
    
    def _get_fallback_generation_result(self, batch_size, device):
        """Fallback 생성 결과"""
        FALLBACK_TEXT = "a panoramic view"
        FALLBACK_TOKEN_COUNT = 5
        
        fallback_text = [FALLBACK_TEXT] * batch_size
        fallback_ids = torch.ones((batch_size, FALLBACK_TOKEN_COUNT), dtype=torch.long, device=device)
        return {"generated_ids": fallback_ids, "text": fallback_text}

    # ==================== 유틸리티 함수들 ====================
    def _project_vision_tokens(self, image_features):
        """비전 특징을 언어모델 임베딩 공간으로 투영"""
        return self.vision_to_language_projection(image_features)
    
    def _compute_manual_cross_entropy_loss(self, logits, labels):
        """
        수동 Cross Entropy Loss 계산 (디버깅/검증용)
        
        중요: HuggingFace CausalLM과 정확히 동일한 시프팅 방식 사용
        - Dataset에서는 시프팅 없이 input_ids와 동일한 위치에 labels 설정
        - 모델 forward에서 내부적으로 next-token prediction을 위한 시프팅 수행
        """
        # HuggingFace CausalLM과 동일한 시프팅 적용
        # logits: (batch, seq_len, vocab_size) -> (batch, seq_len-1, vocab_size)
        # labels: (batch, seq_len) -> (batch, seq_len-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Loss 계산
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 통계 정보
        valid_tokens = (shift_labels != self.ignore_index).sum()
        total_tokens = shift_labels.numel()
        
        # 디버깅 정보 (첫 번째 배치에서만)
        if not hasattr(self, '_debug_shift_logged'):
            print(f"[Loss Debug] Original shapes - logits: {logits.shape}, labels: {labels.shape}")
            print(f"[Loss Debug] Shifted shapes - logits: {shift_logits.shape}, labels: {shift_labels.shape}")
            print(f"[Loss Debug] Valid tokens: {valid_tokens.item()}/{total_tokens} ({valid_tokens.item()/total_tokens*100:.1f}%)")
            print(f"[Loss Debug] Label shifting: input_ids[0->n-1] predicts labels[1->n] (next token)")
            self._debug_shift_logged = True
        
        return {
            "loss": loss,
            "ar_loss": loss,
            "valid_tokens": valid_tokens,
            "total_tokens": total_tokens,
            "perplexity": torch.exp(loss) if torch.isfinite(loss) else torch.tensor(float('inf'))
        }
    
    # ==================== LoRA 지원 메서드들 (선택적) ====================
    
    def setup_lora_for_finetune(self, 
                                lora_r: int = 16, 
                                lora_alpha: int = 32, 
                                lora_dropout: float = 0.1,
                                target_modules: Optional[list] = None) -> bool:
        """
        Finetune 단계에서 LoRA 설정
        
        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: LoRA를 적용할 모듈 이름들 (None이면 기본값 사용)
        
        Returns:
            bool: LoRA 설정 성공 여부
        """
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. LoRA setup skipped.")
            return False
        
        try:
            # 기본 타겟 모듈 (Qwen 모델에 맞춤)
            if target_modules is None:
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # attention layers
                    "gate_proj", "up_proj", "down_proj"      # feed-forward layers
                ]
            
            # LoRA 설정
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # 언어 모델에 LoRA 적용
            self.language_model = get_peft_model(self.language_model, lora_config)
            
            # Gradient checkpointing 호환성 설정
            if hasattr(self.language_model, 'enable_input_require_grads'):
                self.language_model.enable_input_require_grads()
            
            print(f"✓ LoRA setup completed:")
            print(f"  - Rank: {lora_r}")
            print(f"  - Alpha: {lora_alpha}")
            print(f"  - Dropout: {lora_dropout}")
            print(f"  - Target modules: {target_modules}")
            
            # 훈련 가능한 파라미터 수 출력
            trainable_params = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.language_model.parameters())
            print(f"  - Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            return False
    
    
        
    def load_lora_weights(self, load_path: str):
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot load LoRA weights.")
            return False
        try:
            from peft import PeftModel
            is_already_peft = hasattr(self.language_model, 'peft_config')
            if is_already_peft:
                print("Model already has PEFT config. Attempting to load adapter weights...")
                if hasattr(self.language_model, 'load_adapter'):
                    adapter_name = "eval_adapter"
                    try:
                        self.language_model.load_adapter(load_path, adapter_name=adapter_name)
                        self.language_model.set_adapter(adapter_name)
                        print(f"✓ LoRA adapter '{adapter_name}' loaded and activated")
                        return True
                    except Exception as e:
                        print(f"load_adapter failed, fallback to manual state_dict load: {e}")
                import os
                from safetensors.torch import load_file
                adapter_weights_path = os.path.join(load_path, "adapter_model.safetensors")
                if not os.path.exists(adapter_weights_path):
                    print(f"Error: adapter_model.safetensors not found in {load_path}")
                    return False
                state_dict = load_file(adapter_weights_path)
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    ck = k
                    for prefix in ('base_model.model.', 'base_model.'):
                        if ck.startswith(prefix):
                            ck = ck[len(prefix):]
                            break
                    cleaned_state_dict[ck] = v
                missing, unexpected = self.language_model.load_state_dict(cleaned_state_dict, strict=False)
                loaded_cnt = len(cleaned_state_dict) - len(missing)
                if loaded_cnt == 0:
                    print("Error: No LoRA weights matched.")
                    return False
                print(f"✓ LoRA weights loaded: {loaded_cnt}/{len(cleaned_state_dict)} (missing {len(missing)}, unexpected {len(unexpected)})")
                return True
            else:
                print("Converting to PEFT model and loading weights...")
                self.language_model = PeftModel.from_pretrained(self.language_model, load_path, is_trainable=False)
                print("✓ LoRA weights loaded via PeftModel.from_pretrained")
                return True
        except Exception as e:
            import traceback
            print(f"Error loading LoRA weights: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            return False
    
    def merge_lora_weights(self):
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot merge LoRA weights.")
            return False
        if hasattr(self.language_model, 'merge_and_unload'):
            try:
                self.language_model = self.language_model.merge_and_unload()
                print("✓ LoRA weights merged into base model")
                return True
            except Exception as e:
                print(f"Error merging LoRA weights: {e}")
                return False
        else:
            print("Warning: Language model does not support LoRA weight merging.")
            return False

    def get_lora_info(self) -> Dict[str, Any]:
        if not PEFT_AVAILABLE:
            return {"peft_available": False}
        info = {"peft_available": True}
        if hasattr(self.language_model, 'peft_config'):
            peft_config = getattr(self.language_model, 'peft_config', {})
            if peft_config:
                adapter_name = list(peft_config.keys())[0] if peft_config else None
                if adapter_name and adapter_name in peft_config:
                    config = peft_config[adapter_name]
                    info.update({
                        "is_lora_enabled": True,
                        "lora_r": getattr(config, 'r', None),
                        "lora_alpha": getattr(config, 'lora_alpha', None),
                        "lora_dropout": getattr(config, 'lora_dropout', None),
                        "target_modules": getattr(config, 'target_modules', None),
                        "adapter_name": adapter_name
                    })
                else:
                    info["is_lora_enabled"] = False
            else:
                info["is_lora_enabled"] = False
        else:
            info["is_lora_enabled"] = False
        return info

    def save_lora_weights(self, save_path: str) -> bool:
        """
        LoRA 가중치를 지정된 경로에 저장
        
        Args:
            save_path (str): LoRA 가중치를 저장할 디렉토리 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot save LoRA weights.")
            return False
        
        try:
            from pathlib import Path
            
            # LoRA가 활성화되어 있는지 확인
            lora_info = self.get_lora_info()
            if not lora_info.get("is_lora_enabled", False):
                print("Warning: LoRA is not enabled. No weights to save.")
                return False
            
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # PEFT 모델의 어댑터 저장
            # 안전한 저장 유틸리티 사용
            if _safe_save_pretrained(self.language_model, str(save_dir), safe_serialization=True):
                print(f"✓ LoRA weights saved successfully: {save_dir}")
                return True
            else:
                print("Error: Failed to save LoRA weights")
                return False
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ==================== 통합된 모델 로딩 인터페이스 ====================
    
    @classmethod
    def from_checkpoint(cls, 
                       checkpoint_path: str, 
                       lora_weights_path: Optional[str] = None,
                       device: str = "auto",
                       auto_detect_lora: bool = True,
                       strict_loading: bool = False,
                       **model_kwargs) -> 'PanoramaVLM':
        """
        통합된 체크포인트 로딩 메서드 - Lightning 체크포인트와 LoRA 가중치를 자동으로 처리
        
        Args:
            checkpoint_path (str): Lightning 체크포인트 파일 경로 (.ckpt)
            lora_weights_path (str, optional): LoRA 가중치 디렉토리 경로
            device (str): 모델을 로드할 디바이스 ("auto", "cuda", "cpu")
            auto_detect_lora (bool): LoRA 가중치 자동 감지 여부
            strict_loading (bool): 엄격한 가중치 로딩 여부
            **model_kwargs: 모델 생성에 필요한 추가 파라미터들
            
        Returns:
            PanoramaVLM: 로드된 모델 인스턴스
            
        Example:
            # 기본 사용법
            model = PanoramaVLM.from_checkpoint("runs/best.ckpt")
            
            # LoRA 경로 직접 지정
            model = PanoramaVLM.from_checkpoint(
                "runs/best.ckpt", 
                lora_weights_path="runs/lora_weights"
            )
            
            # 모델 파라미터 오버라이드
            model = PanoramaVLM.from_checkpoint(
                "runs/best.ckpt",
                vision_name="google/siglip-large-patch16-384"
            )
        """
        import torch
        from pathlib import Path
        
        print(f"🚀 PanoramaVLM 체크포인트 로딩: {checkpoint_path}")
        
        # 디바이스 설정
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)
        
        # 체크포인트 로드
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        print(f"📂 체크포인트 로딩 중...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device_obj)
        except Exception as e:
            raise RuntimeError(f"체크포인트 로딩 실패: {e}")
        
        # 하이퍼파라미터 추출
        hparams = checkpoint.get('hyper_parameters', {})
        model_state_dict = checkpoint.get('state_dict', {})
        
        # 설정 시스템을 활용한 모델 파라미터 결정
        try:
            # 1. 설정 파일 자동 감지 시도
            from .config import ConfigManager, ModelConfig
            detected_config = ConfigManager.auto_detect_config(checkpoint_path)
            
            if detected_config:
                print(f"🔍 설정 파일 자동 감지 성공")
                model_config = detected_config
            else:
                print(f"🔍 설정 파일 감지 실패 - 하이퍼파라미터에서 생성")
                # 2. 하이퍼파라미터에서 설정 생성
                config_dict = {
                    'vision_name': hparams.get('vision_name', 'google/siglip-base-patch16-224'),
                    'language_model_name': hparams.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'),
                    'resampler_type': hparams.get('resampler_type', 'mlp'),
                    'latent_dimension': hparams.get('latent_dimension', 768),
                    'vicreg_loss_weight': hparams.get('vicreg_loss_weight', 1.0),
                    'vicreg_overlap_ratio': hparams.get('vicreg_overlap_ratio', 0.5),
                    'max_text_length': hparams.get('max_text_length', 512),
                }
                model_config = ModelConfig.from_dict(config_dict)
            
            # 3. 사용자 지정 파라미터로 오버라이드
            if model_kwargs:
                print(f"🛠️  사용자 파라미터로 설정 오버라이드: {list(model_kwargs.keys())}")
                model_config = model_config.update(**model_kwargs)
            
            # 4. 모델 생성용 파라미터 추출
            model_params = model_config.get_model_kwargs()
            model_params['config'] = model_config  # config 객체도 전달
            
        except Exception as e:
            print(f"⚠️ 설정 시스템 사용 실패 ({e}) - 기존 방식 사용")
            # 폴백: 기존 방식
            model_params = {
                'vision_name': 'google/siglip-base-patch16-224',
                'language_model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
                'resampler_type': 'mlp',
                'latent_dimension': 768,
                'vicreg_loss_weight': 1.0,
                'vicreg_overlap_ratio': 0.5,
                'max_text_length': 512,
            }
            
            # 하이퍼파라미터에서 업데이트
            for key in model_params.keys():
                if key in hparams:
                    model_params[key] = hparams[key]
            
            # 사용자 지정 파라미터로 최종 업데이트
            model_params.update(model_kwargs)
        
        print(f"🛠️  모델 파라미터:")
        for key, value in model_params.items():
            print(f"   - {key}: {value}")
        
        # 모델 인스턴스 생성
        print(f"🏗️  모델 인스턴스 생성 중...")
        model = cls(**model_params)
        
        # Lightning wrapper에서 실제 모델 가중치 추출
        print(f"⚙️  가중치 로딩 중...")
        model_weights = {}
        for key, value in model_state_dict.items():
            if key.startswith('model.'):
                # 'model.' 접두어 제거
                clean_key = key[6:]  # len('model.') = 6
                model_weights[clean_key] = value
        
        # 가중치 로드
        if model_weights:
            missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=strict_loading)
            print(f"   - 로드된 키: {len(model_weights) - len(missing_keys)}")
            if missing_keys:
                print(f"   - 누락된 키: {len(missing_keys)} ({missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''})")
            if unexpected_keys:
                print(f"   - 예상치 못한 키: {len(unexpected_keys)} ({unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''})")
        else:
            print("   ⚠️  모델 가중치를 찾을 수 없습니다. 기본 초기화된 모델을 사용합니다.")
        
        # LoRA 가중치 처리
        if auto_detect_lora and lora_weights_path is None:
            # 자동 감지: 체크포인트와 같은 디렉토리에서 lora_weights 폴더 찾기
            checkpoint_dir = checkpoint_path.parent
            potential_lora_path = checkpoint_dir / "lora_weights"
            if potential_lora_path.exists() and potential_lora_path.is_dir():
                lora_weights_path = str(potential_lora_path)
                print(f"🔍 LoRA 가중치 자동 감지: {lora_weights_path}")
        
        if lora_weights_path:
            lora_path = Path(lora_weights_path)
            if lora_path.exists():
                print(f"🔧 LoRA 가중치 로딩: {lora_weights_path}")
                success = model.load_lora_weights(str(lora_path))
                if success:
                    lora_info = model.get_lora_info()
                    if lora_info.get("is_lora_enabled", False):
                        print(f"   ✅ LoRA 로딩 성공 - Rank: {lora_info.get('lora_r')}, Alpha: {lora_info.get('lora_alpha')}")
                    else:
                        print(f"   ⚠️  LoRA 상태 확인 실패")
                else:
                    print(f"   ❌ LoRA 로딩 실패")
            else:
                print(f"   ⚠️  LoRA 경로가 존재하지 않습니다: {lora_weights_path}")
        
        # 모델을 지정된 디바이스로 이동
        model = model.to(device_obj)
        model.eval()  # 기본적으로 평가 모드
        
        # 토크나이저 정보 추가 (eval.py 호환성)
        if not hasattr(model, 'tokenizer'):
            try:
                from transformers import AutoTokenizer
                tokenizer_name = model_params.get('language_model_name', 'Qwen/Qwen3-0.6B')
                model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"   ✅ 토크나이저 로드: {tokenizer_name}")
            except Exception as e:
                print(f"   ⚠️ 토크나이저 로드 실패: {e}")
        
        print(f"✅ 모델 로딩 완료 - Device: {device}")
        return model
    
    @classmethod  
    def from_pretrained(cls, 
                       model_path: str,
                       device: str = "auto", 
                       **kwargs) -> 'PanoramaVLM':
        """
        HuggingFace 스타일 인터페이스 - 사전 훈련된 모델 로딩
        
        Args:
            model_path (str): 모델 경로 (체크포인트 파일 또는 디렉토리)
            device (str): 디바이스 설정
            **kwargs: 추가 모델 파라미터들
            
        Returns:
            PanoramaVLM: 로드된 모델
            
        Example:
            model = PanoramaVLM.from_pretrained("runs/panorama-vlm-e2p")
            model = PanoramaVLM.from_pretrained("runs/best.ckpt")
        """
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # 디렉토리인 경우 best.ckpt 또는 model_final.safetensors 찾기
        if model_path.is_dir():
            checkpoint_candidates = [
                model_path / "best.ckpt",
                model_path / "last.ckpt", 
                model_path / "model.safetensors",
                model_path / "model_final.safetensors",
                model_path / "pytorch_model.bin"
            ]
            
            checkpoint_path = None
            for candidate in checkpoint_candidates:
                if candidate.exists():
                    checkpoint_path = candidate
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(f"지원되는 모델 파일을 찾을 수 없습니다: {model_path}")
        else:
            checkpoint_path = model_path
        
        return cls.from_checkpoint(str(checkpoint_path), device=device, **kwargs)
    
    def save_pretrained(self, save_directory: str, save_lora_separately: bool = True):
        """
        HuggingFace 스타일 모델 저장
        
        Args:
            save_directory (str): 저장할 디렉토리 경로
            save_lora_separately (bool): LoRA 가중치를 별도로 저장할지 여부
        """
        from pathlib import Path
        import torch
        import json
        
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 모델 저장 중: {save_directory}")
        
        # 모델 가중치 저장 (SafeTensors 형식)
        try:
            from safetensors.torch import save_file
            model_path = save_dir / "model.safetensors"
            save_file(self.state_dict(), model_path)
            print(f"   ✅ 모델 가중치 저장 (SafeTensors): {model_path}")
        except ImportError:
            print("   ⚠️ SafeTensors 라이브러리가 없어 PyTorch 형식으로 저장합니다")
            model_path = save_dir / "pytorch_model.bin" 
            torch.save(self.state_dict(), model_path)
            print(f"   ✅ 모델 가중치 저장 (PyTorch): {model_path}")
        
        # 설정 정보 저장 (ModelConfig 시스템 사용)
        try:
            # 현재 모델의 설정을 업데이트된 정보로 갱신
            if hasattr(self, 'config') and self.config:
                updated_config = self.config.update(
                    vision_name=getattr(self.vision_encoder.config, 'name_or_path', self.config.vision_name),
                    language_model_name=getattr(self.language_model.config, 'name_or_path', self.config.language_model_name),
                    latent_dimension=self.vision_to_language_projection.in_features,
                    max_text_length=self.max_text_length,
                    vicreg_loss_weight=self.vicreg_loss_weight,
                    vicreg_overlap_ratio=self.vicreg_overlap_ratio,
                    description=f"Saved at {save_directory}"
                )
            else:
                # config가 없는 경우 새로 생성
                from .config import ModelConfig
                updated_config = ModelConfig(
                    vision_name=getattr(self.vision_encoder.config, 'name_or_path', 'unknown'),
                    language_model_name=getattr(self.language_model.config, 'name_or_path', 'unknown'),
                    latent_dimension=self.vision_to_language_projection.in_features,
                    max_text_length=self.max_text_length,
                    vicreg_loss_weight=self.vicreg_loss_weight,
                    vicreg_overlap_ratio=self.vicreg_overlap_ratio,
                    description=f"Saved at {save_directory}"
                )
            
            # ModelConfig 형식으로 저장
            config_path = save_dir / "model_config.json"
            updated_config.save(config_path)
            print(f"   ✅ ModelConfig 저장: {config_path}")
            
            # 하위 호환성을 위한 기존 형식도 저장
            legacy_config = {
                "model_type": "PanoramaVLM",
                "vision_name": updated_config.vision_name,
                "language_model_name": updated_config.language_model_name,
                "latent_dimension": updated_config.latent_dimension,
                "max_text_length": updated_config.max_text_length,
                "vicreg_loss_weight": updated_config.vicreg_loss_weight,
                "vicreg_overlap_ratio": updated_config.vicreg_overlap_ratio,
            }
            
            legacy_config_path = save_dir / "config.json"
            with open(legacy_config_path, 'w', encoding='utf-8') as f:
                json.dump(legacy_config, f, indent=2, ensure_ascii=False)
            print(f"   ✅ 레거시 설정 저장: {legacy_config_path}")
            
        except Exception as e:
            print(f"   ⚠️ ModelConfig 저장 실패 ({e}) - 기존 방식 사용")
            # 폴백: 기존 방식
            config = {
                "model_type": "PanoramaVLM",
                "vision_name": getattr(self.vision_encoder.config, 'name_or_path', 'unknown'),
                "language_model_name": getattr(self.language_model.config, 'name_or_path', 'unknown'),
                "latent_dimension": self.vision_to_language_projection.in_features,
                "max_text_length": self.max_text_length,
                "vicreg_loss_weight": self.vicreg_loss_weight,
                "vicreg_overlap_ratio": self.vicreg_overlap_ratio,
            }
            
            config_path = save_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"   ✅ 설정 저장: {config_path}")
        
        # LoRA 가중치 별도 저장
        if save_lora_separately:
            lora_info = self.get_lora_info()
            if lora_info.get("is_lora_enabled", False):
                lora_dir = save_dir / "lora_weights"
                success = self.save_lora_weights(str(lora_dir))
                if success:
                    print(f"   ✅ LoRA 가중치 저장: {lora_dir}")
                else:
                    print(f"   ⚠️ LoRA 가중치 저장 실패: {lora_dir}")
        
        print(f"🎉 모델 저장 완료: {save_directory}")
    
    @staticmethod
    def create_model_factory(checkpoint_path: str, **default_kwargs):
        """
        모델 팩토리 함수 생성 - 반복적인 로딩을 위한 편의 함수
        
        Args:
            checkpoint_path (str): 기본 체크포인트 경로
            **default_kwargs: 기본 모델 파라미터들
            
        Returns:
            function: 모델 생성 함수
            
        Example:
            # 팩토리 생성
            model_factory = PanoramaVLM.create_model_factory(
                "runs/best.ckpt",
                device="cuda:0"
            )
            
            # 모델 생성
            model1 = model_factory()
            model2 = model_factory(max_text_length=256)
        """
        def factory(**kwargs):
            # 기본값과 사용자 인자 병합
            merged_kwargs = {**default_kwargs, **kwargs}
            return PanoramaVLM.from_checkpoint(checkpoint_path, **merged_kwargs)
        
        return factory