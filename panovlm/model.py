# coding: utf-8

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .utils import set_seed, safe_save_pretrained, infer_hw
from .losses import VicRegLoss
from .resampler.resamplers import MLPResampler, ConvResampler

# LoRA 지원을 위한 PEFT import (선택적)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA fine-tuning will not be supported.")

# ==================== Utilities imported from panovlm.utils ====================

# ---------------------------------------------------------------------------
# ‣ Panorama-specific Positional Encoding
# ---------------------------------------------------------------------------
class PanoramaPositionalEncoding(nn.Module):
    """
    Panorama-aware positional encoding that enforces yaw continuity across views.

    - Input: resampled vision tokens with shape [B*V, S, D]
      where B=batch, V=views, S=spatial tokens (H*W), D=embed dim.
    - Output: same shape, with sinusoidal view+spatial PE added.

    Key ideas:
    - Yaw continuity: Treat each view index as an angle phi in [0, 2π).
      Use sin/cos features so that the first and last views are adjacent.
    - Spatial PE: Standard 2D sinusoidal PE on (row, col) grid and additively combined.
    - No assumptions about overlap: Whether crops overlap or not, yaw PE stays cyclic.
    """

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
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.view_encoding_type = view_encoding_type
        self.spatial_encoding_type = spatial_encoding_type
        self.enable_continuity = bool(enable_continuity)
        self.temperature = float(temperature)
        # fraction of horizontal overlap between adjacent views, e.g., 0.5 for 50%
        self.overlap_ratio = max(0.0, min(float(overlap_ratio), 0.999))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    @staticmethod
    def _build_sinusoidal(pos: torch.Tensor, dim: int, temperature: float, dtype: torch.dtype) -> torch.Tensor:
        """
        Standard sinusoidal embedding for given positions (float tensor).
        pos: [...]
        returns: [..., dim]
        """
        device = pos.device
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        pos_f = pos.to(compute_dtype)
        half = dim // 2
        if half == 0:
            return torch.zeros(*pos.shape, dim, device=device, dtype=dtype)
        # frequencies: 1 / temperature^(2i/d)
        idx = torch.arange(half, device=device, dtype=compute_dtype)
        div = torch.exp(-math.log(temperature) * (2 * idx / max(1, dim)))
        # broadcast pos to [..., 1]
        ang = pos_f.unsqueeze(-1) * div  # [..., half]
        emb_sin = torch.sin(ang)
        emb_cos = torch.cos(ang)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # [..., dim or dim-1]
        if emb.shape[-1] < dim:
            # pad one zero if odd dim
            pad = torch.zeros(*emb.shape[:-1], dim - emb.shape[-1], device=device, dtype=compute_dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb.to(dtype)

    def _yaw_encoding(self, num_views: int, batch_size: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Yaw encoding; if overlap is specified, use global continuous x to align overlaps.

        Returns tensor of shape [B, V, H, W, D]
        """
        V, D = num_views, self.embed_dim
        s = 1.0 - self.overlap_ratio  # stride in view-width units
        if not self.enable_continuity:
            s = 1.0

        base_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        v_idx = torch.arange(V, device=device, dtype=base_dtype).view(V, 1)  # [V,1]
        x = torch.arange(W, device=device, dtype=base_dtype) / max(1.0, float(W))  # [W]
        # global position along panorama in [0, L_total]
        g = v_idx * s + x.unsqueeze(0)  # [V, W]
        L_total = V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)
        # map to angle domain (not strictly necessary but keeps period explicit)
        phi = (2.0 * math.pi) * (g / max(1e-6, float(L_total)))  # [V, W]

        if self.view_encoding_type != "sinusoidal":
            # default to sinusoidal
            pass
        yaw_vw = self._build_sinusoidal(phi, D, self.temperature, base_dtype)  # [V, W, D]
        # expand across rows H and batch B
        yaw = yaw_vw.view(1, V, 1, W, D).expand(batch_size, V, H, W, D)
        return yaw.to(dtype)

    def _spatial_encoding(self, H: int, W: int, V: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.spatial_encoding_type != "sinusoidal":
            # default to zeros if disabled/unknown
            return torch.zeros(V, H, W, self.embed_dim, device=device, dtype=dtype)

        # Row (y) is local to each view and naturally aligns across overlaps
        base_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        y = torch.arange(H, device=device, dtype=base_dtype)
        y_emb = self._build_sinusoidal(y, self.embed_dim, self.temperature, base_dtype)  # [H, D]

        # Column (x) uses global panorama coordinate when overlap is enabled
        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        v_idx = torch.arange(V, device=device, dtype=base_dtype).view(V, 1)
        x_local = torch.arange(W, device=device, dtype=base_dtype) / max(1.0, float(W))  # [W]
        g = v_idx * s + x_local.unsqueeze(0)  # [V, W]
        L_total = V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)
        # Use g (not necessarily angle) as position input for sinusoidal
        x_emb = self._build_sinusoidal(g, self.embed_dim, self.temperature, base_dtype)  # [V, W, D]

        # Combine: for each view v, grid[y, x] = y_emb[y] + x_emb[v, x]
        grid = y_emb.view(1, H, 1, self.embed_dim) + x_emb.view(V, 1, W, self.embed_dim)
        return grid.to(dtype)  # [V, H, W, D]

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        # x: [B*V, S, D]
        BV, S, D = x.shape
        assert D == self.embed_dim, f"Embed dim mismatch: x={D}, pe={self.embed_dim}"
        H, W = infer_hw(S)

        device = x.device
        # [B, V, H, W, D]
        xv = x.view(batch_size, num_views, H, W, D)

        # Build PE components (yaw and spatial use global panorama x when overlap_ratio>0)
        yaw_pe = self._yaw_encoding(num_views=num_views, batch_size=batch_size, H=H, W=W, device=device, dtype=x.dtype)  # [B,V,H,W,D]
        spat_pe = self._spatial_encoding(H, W, num_views, device, x.dtype)  # [1,V,H,W,D]

        pe = yaw_pe + spat_pe  # [B,V,H,W,D]
        out = (xv + pe).view(BV, S, D)
        return self.dropout(out)

# ---------------------------------------------------------------------------
# ‣ Panorama-specific Positional Encoding (Spherical 3D PE, NeRF-style)
# ---------------------------------------------------------------------------
class PanoramaPositionalEncoding2(nn.Module):
    """
    Spherical 3D positional encoding for panoramic tokens.

    변경 사항 (요약):
    - 기존 phi(수평) 1D + 로컬 2D sinusoidal → (x,y,z) 구면 좌표 기반 멀티주파수 Fourier PE(NeRF-style).
    - 수평(경도, φ)은 뷰 인덱스 + 오버랩 비율을 고려한 전역 x축으로 연속화.
    - 수직(위도, θ)은 ERP 가정으로 H축을 [-π/2, +π/2] 범위에 매핑(옵션으로 중심/범위 조정).
    - 생성된 [B,V,H,W,raw_dim] PE를 선형사상해 최종 임베드 차원(embed_dim)으로 맞춘 뒤 입력 토큰에 더함.

    입력:
        x: [B*V, S, D],  S=H*W (패치/토큰 그리드)
    출력:
        [B*V, S, D]  (동일 차원, add)
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        # ---- 기존 인자 유지(하위호환) ----
        view_encoding_type: str = "sinusoidal",      # [DEPRECATED] 무시됨
        spatial_encoding_type: str = "sinusoidal",   # [DEPRECATED] 무시됨
        enable_continuity: bool = True,
        overlap_ratio: float = 0.0,
        temperature: float = 10000.0,                # [DEPRECATED] 무시됨
        dropout: float = 0.0,
        # ---- 새 인자 (구면 3D-PE 제어) ----
        num_fourier_bands: int = 8,                  # 주파수 밴드 개수 (NeRF-style, 2^l 스케일)
        include_input_xyz: bool = True,              # sin/cos 이전의 원시 (x,y,z) 포함 여부
        pe_scale: float = math.pi,                   # sin/cos 인자 스케일(기본 π)
        phi_offset_rad: float = 0.0,                 # 수평 회전 오프셋(라디안)
        lat_center_rad: float = 0.0,                 # 수직 중심 위도(기본 적도=0)
        lat_coverage_ratio: float = 1.0,             # 수직 커버리지 비율(1.0이면 전체 [-π/2,+π/2])
        project_bias: bool = True,                   # 최종 선형사상 바이어스 사용
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.enable_continuity = bool(enable_continuity)
        self.overlap_ratio = float(max(0.0, min(overlap_ratio, 0.999)))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # Spherical PE 설정
        self.num_fourier_bands = int(num_fourier_bands)
        self.include_input_xyz = bool(include_input_xyz)
        self.pe_scale = float(pe_scale)
        self.phi_offset_rad = float(phi_offset_rad)

        # 수직(위도) 매핑 파라미터
        # 기본: θ(y) ∈ [-π/2, +π/2], 여기에 중심(lat_center)과 커버리지(coverage)로 미세조정
        self.lat_center_rad = float(lat_center_rad)
        self.lat_coverage_ratio = float(max(1e-6, min(lat_coverage_ratio, 1.0)))

        # 원시 3D-PE의 채널 수 계산 (include xyz + sin/cos)
        self._raw_dim = self._compute_raw_dim()
        self._proj = nn.Linear(self._raw_dim, self.embed_dim, bias=bool(project_bias))

    # --------------------------- 내부 유틸리티 ---------------------------

    def _compute_raw_dim(self) -> int:
        """원시 (x,y,z) + Fourier(sin/cos) 특징 차원 계산."""
        C = 3  # x,y,z
        fourier_dim = C * (2 * self.num_fourier_bands)  # sin+cos
        base_dim = C if self.include_input_xyz else 0
        return base_dim + fourier_dim

    @staticmethod
    def _infer_hw(S: int) -> Tuple[int, int]:
        # 안전한 그리드 추정(사용자 util이 있으면 그걸 사용해도 됨)
        r = int(math.sqrt(S))
        if r * r == S:
            return r, r
        # 가장 가까운 약수 조합 탐색(보수적)
        for h in range(r, 1, -1):
            if S % h == 0:
                return h, S // h
        return 1, S

    @staticmethod
    def _spherical_xyz(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        θ: 위도 in [-π/2, +π/2],  φ: 경도 in [-π, +π] (또는 [0, 2π], 일관성만 유지)
        반환: [..., 3] with (x, y, z) on unit sphere
        """
        ct = torch.cos(theta)
        x = ct * torch.cos(phi)
        y = torch.sin(theta)
        z = ct * torch.sin(phi)
        return torch.stack([x, y, z], dim=-1)

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [..., 3]  (x,y,z)
        반환: [..., 3 * (2*num_bands)]  with [sin(2^l*scale*c), cos(2^l*scale*c)]
        """
        device = coords.device
        BANDS = self.num_fourier_bands
        if BANDS <= 0:
            return coords.new_zeros(*coords.shape[:-1], 0)

        # [num_bands] 주파수 (2^l * pe_scale)
        freq = (2.0 ** torch.arange(BANDS, device=device, dtype=coords.dtype)) * self.pe_scale  # [BANDS]
        # [..., 3, 1] * [BANDS] -> [..., 3, BANDS]
        ang = coords.unsqueeze(-1) * freq  # [..., 3, BANDS]
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        out = torch.cat([sin, cos], dim=-1)  # [..., 3, 2*BANDS]
        return out.view(*coords.shape[:-1], 3 * (2 * BANDS))

    def _global_longitude(self, V: int, W: int, device: torch.device) -> torch.Tensor:
        """
        오버랩 고려 전역 수평 좌표 → 경도 φ (라디안)
        - view 간 stride s = 1 - overlap_ratio
        - 전체 길이 L_total = V - (V-1)*overlap
        - g ∈ [0, L_total], φ = 2π * (g / L_total) + phi_offset
        반환: [V, W] (각 뷰 v의 칼럼 x에 대한 φ)
        """
        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        v_idx = torch.arange(V, device=device, dtype=torch.float32).view(V, 1)    # [V,1]
        x = torch.arange(W, device=device, dtype=torch.float32) / max(1.0, float(W))  # [W] in [0,1)
        g = v_idx * s + x.unsqueeze(0)  # [V, W]
        L_total = V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)
        L_total = max(float(L_total), 1e-6)
        phi = (2.0 * math.pi) * (g / L_total) + self.phi_offset_rad
        return phi

    def _latitude_from_rows(self, H: int, device: torch.device) -> torch.Tensor:
        """
        H행을 ERP 가정으로 θ(y) ∈ [-π/2, +π/2]에 매핑 후,
        중심/범위(lat_center, coverage)로 재조정.
        반환: [H]
        """
        # y ∈ [0, H-1] → u ∈ [0,1] → θ_raw ∈ [-π/2, +π/2]
        y = torch.arange(H, device=device, dtype=torch.float32)
        u = (y + 0.5) / max(1.0, float(H))  # 픽셀 중심 기준
        theta_raw = (u - 0.5) * math.pi  # [-π/2, +π/2]
        # 커버리지(0<r≤1): θ = center + r * theta_raw
        theta = self.lat_center_rad + (self.lat_coverage_ratio * theta_raw)
        return theta

    # ------------------------------ Forward ------------------------------

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        # x: [B*V, S, D]
        BV, S, D = x.shape
        assert D == self.embed_dim, f"Embed dim mismatch: x={D}, pe={self.embed_dim}"
        H, W = self._infer_hw(S)
        device = x.device

        # 입력을 [B,V,H,W,D]로 복원
        xv = x.view(batch_size, num_views, H, W, D)

        # 1) 경도 φ[v, w] (전역 연속), 2) 위도 θ[h]
        phi_vw = self._global_longitude(num_views, W, device)          # [V, W]
        theta_h = self._latitude_from_rows(H, device)                  # [H]

        # 2D 그리드 확장 → [B,V,H,W]
        phi = phi_vw.view(1, num_views, 1, W).expand(batch_size, num_views, H, W)
        theta = theta_h.view(1, 1, H, 1).expand(batch_size, num_views, H, W)

        # (x,y,z) 구면 좌표
        xyz = self._spherical_xyz(theta, phi)  # [B,V,H,W,3]

        # NeRF-style Fourier features
        feats = []
        if self.include_input_xyz:
            feats.append(xyz)
        feats.append(self._fourier_encode(xyz))  # [B,V,H,W, 3*(2*BANDS)]
        pe_raw = torch.cat(feats, dim=-1)        # [B,V,H,W, raw_dim]

        # 선형 사상으로 embed_dim에 맞춤
        pe = self._proj(pe_raw)                  # [B,V,H,W,D]
        out = xv + pe
        out = out.view(BV, S, D)
        return self.dropout(out)



# ---------------------------------------------------------------------------
# ‣ Simple MLP Blocks (moved MLPResampler to panovlm/resampler/resamplers.py)
# ---------------------------------------------------------------------------

class VICRegProjector(nn.Module):
    """VICReg 전용 Projector: token-wise MLP (in→h→out), depth=2~3 권장"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, depth: int = 2, use_ln: bool = True):
        super().__init__()
        hd = hidden_dim or max(in_dim, out_dim)
        layers = []
        d_prev = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_prev, hd))
            layers.append(nn.LayerNorm(hd) if use_ln else nn.Identity())
            layers.append(nn.GELU())
            d_prev = hd
        layers.append(nn.Linear(d_prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*V, S, D]
        BVS, S, D = x.shape
        x = x.reshape(-1, D)
        x = self.mlp(x)
        return x.view(BVS, S, -1)

# ---------------------------------------------------------------------------
# ‣ VICReg Loss (moved to panovlm/utils/losses.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ‣ PanoramaVLM (VICReg + AR)
# ---------------------------------------------------------------------------
class PanoramaVLM(nn.Module):
    """
    3단계 학습:
    1) stage="vision": VICReg (인접 뷰 겹치는 패치 구간만)
       - NEW: VICReg projector로 차원 축소/정규화 후 손실 계산
    2) stage="resampler": Resampler + LM AR-loss (resampler와 proj만 학습 가능)
    3) stage="finetune": 전체 미세조정
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()

        # 설정 ------------------------------------------------
        if config is not None:
            self.config = config
        else:
            # 간단한 dict 기반 설정 폴백
            class _Cfg:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)
            self.config = _Cfg(**kwargs)

        # 비전 인코더 ------------------------------------------
        self.vision_encoder = AutoModel.from_pretrained(
            getattr(self.config, 'vision_name', 'google/siglip-base-patch16-224'),
            trust_remote_code=True
        )
        if hasattr(self.vision_encoder, "vision_model"):
            self.vision_encoder = self.vision_encoder.vision_model
        vision_hidden_size = self._get_vision_hidden_size(self.vision_encoder)

        # VICReg 정규화 레이어 (옵션)
        self.use_vicreg_norm = getattr(self.config, 'use_vicreg_norm', False)
        self.vicreg_norm = nn.LayerNorm(vision_hidden_size) if self.use_vicreg_norm else nn.Identity()

        # NEW: VICReg 전용 Projector ---------------------------
        self.use_vicreg_projector = getattr(self.config, 'use_vicreg_projector', True)
        self.vicreg_projector_dim = int(getattr(self.config, 'vicreg_projector_dim', 768))
        self.vicreg_projector_depth = int(getattr(self.config, 'vicreg_projector_depth', 2))
        self.vicreg_projector_hidden = int(getattr(self.config, 'vicreg_projector_hidden', self.vicreg_projector_dim))
        self.vicreg_projector_ln = bool(getattr(self.config, 'vicreg_projector_ln', True))

        if self.use_vicreg_projector:
            self.vicreg_projector = VICRegProjector(
                in_dim=vision_hidden_size,
                out_dim=self.vicreg_projector_dim,
                hidden_dim=self.vicreg_projector_hidden,
                depth=self.vicreg_projector_depth,
                use_ln=self.vicreg_projector_ln,
            )
            self._vicreg_feat_dim = self.vicreg_projector_dim
        else:
            self.vicreg_projector = nn.Identity()
            self._vicreg_feat_dim = vision_hidden_size

        # 리샘플러 ----------------------------------------------
        resampler_type = getattr(self.config, 'resampler_type', 'mlp')
        latent_dimension = int(getattr(self.config, 'latent_dimension', vision_hidden_size))
        resampler_depth = getattr(self.config, 'resampler_depth', 2)
        resampler_hidden_dim = getattr(self.config, 'resampler_hidden_dim', None)
        resampler_use_ln = getattr(self.config, 'resampler_use_ln', True)
        
        if resampler_type == "mlp":
            self.resampler = MLPResampler(
                vision_hidden_size, 
                latent_dimension,
                hidden_dim=resampler_hidden_dim,
                depth=resampler_depth,
                use_ln=resampler_use_ln
            )
        else:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {resampler_type}")

        # 프로젝션 단계 Positional Encoding (파노라마 인지)
        # 기본값: sinusoidal view+spatial, 연속성 강화 on, dropout 0.0
        self.use_projection_pe = bool(getattr(self.config, 'use_projection_positional_encoding', True))
        try:
            # Always tie PE overlap to VICReg overlap for consistent training signal.
            vicreg_or_default = float(getattr(self.config, 'overlap_ratio', 0.5))
            pe_kwargs = dict(
                embed_dim=latent_dimension,
                view_encoding_type=getattr(self.config, 'pe_view_encoding_type', 'sinusoidal'),
                spatial_encoding_type=getattr(self.config, 'pe_spatial_encoding_type', 'sinusoidal'),
                enable_continuity=bool(getattr(self.config, 'pe_enable_continuity', True)),
                overlap_ratio=vicreg_or_default,
                temperature=float(getattr(self.config, 'pe_temperature', 10000.0)),
                dropout=float(getattr(self.config, 'pe_dropout', 0.0)),
            )
        except Exception:
            pe_kwargs = dict(embed_dim=latent_dimension)
        self.projection_pe = PanoramaPositionalEncoding(**pe_kwargs) if self.use_projection_pe else None

        # 언어 모델 및 투영 ------------------------------------
        lm_name = getattr(self.config, 'language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
        self.language_model = AutoModelForCausalLM.from_pretrained(
            lm_name,
            attn_implementation="sdpa",
        )
        # Reduce GPU footprint during training
        if hasattr(self.language_model, "config"):
            self.language_model.config.use_cache = False
        self.vision_to_language_projection = nn.Linear(latent_dimension, self.language_model.config.hidden_size)
        self._gradient_checkpointing = False

        # 토크나이저 -------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self._setup_tokenizer()
        self.vision_token_id = self.tokenizer.convert_tokens_to_ids("<|vision|>")
        if self.vision_token_id == self.tokenizer.unk_token_id:
            self.vision_token_id = self.tokenizer.bos_token_id

        # VICReg 손실 및 하이퍼 --------------------------------
        self.vicreg_loss = VicRegLoss(
            similarity_weight=float(getattr(self.config, 'vicreg_similarity_weight', 25.0)),
            variance_weight=float(getattr(self.config, 'vicreg_variance_weight', 25.0)),
            covariance_weight=float(getattr(self.config, 'vicreg_covariance_weight', 1.0)),
            gamma=float(getattr(self.config, 'vicreg_gamma', 1.0)),
            use_ddp_gather=bool(getattr(self.config, 'vicreg_use_ddp_gather', False)),
        )
        self.vicreg_loss_weight = float(getattr(self.config, 'vicreg_loss_weight', 1.0))
        self.overlap_ratio = float(getattr(self.config, 'overlap_ratio', 0.5))

        # 토큰 결합 방식 (중복 제거 전략)
        # - 'drop_overlap': 기존 방식. 첫 뷰는 전체, 이후 뷰는 좌측 k=round(W*overlap) 열을 드랍 후 이어붙임
        # - 'stride_views': 겹침이 0이 되도록 뷰 인덱스를 간격 s=ceil(1/(1-overlap))로 샘플링 (예: 50%면 0,2,4,...)
        # - 'concat': 단순 뷰-인터리브(중복 제거 안 함)
        # - 'resample': 파노라마 전역 좌표로 재표본화하여 목표 가로 토큰 수로 정규화
        self.stitching_mode = str(getattr(self.config, 'stitching_mode', 'stride_views'))
        self.stitch_stride_offset = int(getattr(self.config, 'stitch_stride_offset', 0))
        self.stitch_target_cols = int(getattr(self.config, 'stitch_target_cols', 0))
        self.stitch_target_to_view_width = bool(getattr(self.config, 'stitch_target_to_view_width', False))
        self.stitch_interp = str(getattr(self.config, 'stitch_interp', 'nearest'))

        # 텍스트 설정 ------------------------------------------
        self.max_text_length = int(getattr(self.config, 'max_text_length', 512))
        self.ignore_index = -100

        # 디버그 플래그
        self._warned_single_view = False
        self._debug_loss_verification = False
        self._cached_module_dtypes: Dict[str, torch.dtype] = {}

    # ---------------- 유틸리티 ---------------------------------------------
    @staticmethod
    def _get_vision_hidden_size(vision_model: nn.Module) -> int:
        for key in ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]:
            if hasattr(vision_model.config, key):
                return getattr(vision_model.config, key)
        raise AttributeError("비전 모델의 은닉 차원 크기를 찾을 수 없습니다")

    def _has_cls_token(self, vision_output: torch.Tensor) -> bool:
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

    def _setup_tokenizer(self):
        tokens_added = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[Tokenizer Setup] Set pad_token = eos_token: '{self.tokenizer.eos_token}'")
            else:
                new_tokens = {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}
                self.tokenizer.add_special_tokens(new_tokens)
                tokens_added = True
                print(f"[Tokenizer Setup] Added eos_token and pad_token: '<|endoftext|>'")
        special_tokens_to_add = []
        vision_token = '<|vision|>'
        if not any(vision_token in str(token) for token in self.tokenizer.additional_special_tokens):
            special_tokens_to_add.append(vision_token)
        if self.tokenizer.eos_token != '<|endoftext|>':
            if not any('<|endoftext|>' in str(token) for token in self.tokenizer.additional_special_tokens):
                special_tokens_to_add.append('<|endoftext|>')
        if special_tokens_to_add:
            added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
            if added_tokens > 0:
                tokens_added = True
                print(f"[Tokenizer Setup] Added {added_tokens} special tokens: {special_tokens_to_add}")
        if tokens_added:
            old_embeddings = self.language_model.get_input_embeddings().weight.size(0)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            new_embeddings = self.language_model.get_input_embeddings().weight.size(0)
            print(f"[Tokenizer Setup] Resized embeddings: {old_embeddings} -> {new_embeddings}")
        self.tokenizer.padding_side = "right"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        if self.pad_token_id == self.eos_token_id:
            print(f"[Tokenizer Setup] ✓ pad_token_id == eos_token_id (consistent with loss masking)")

    # ==================== 공통 처리 함수들 ====================
    def _process_vision_encoder(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
        """
        이미지 B,V,3,H,W를 B*V,3,H,W로 변환하여 비전 인코더를 통과시킴
        
        Returns:
            Dict with vision_features: [B*V, S, D_vision], batch_size, num_views
        """
        batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
        vision_features = self._extract_vision_features(normalized_pixels, batch_size, num_views)  # [B*V, S, D_vision]
        
        return {
            "vision_features": vision_features,
            "batch_size": batch_size,
            "num_views": num_views,
            "device": normalized_pixels.device
        }
    
    def _process_resampler(self, vision_features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Resampler를 통한 vision feature 변환

        Args:
            vision_features: [B*V, S, D_vision] - Vision encoder의 출력

        Returns:
            resampled_features: [B*V, S, D_latent] - Resampler 출력
        """
        target_dtype = self._get_module_dtype("resampler", self.resampler, vision_features.dtype)
        if target_dtype is not None and vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        return self.resampler(vision_features)  # [B*V, S, D_latent]
    
    def _process_projection_layer(self, resampled_features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Projection layer 처리: B*V,L,D -> B,V*L,D 변환
        
        Args:
            resampled_features: [B*V, S, D_latent] - Resampler 출력
            
        Returns:
            projected_features: [B, V*S, D_lm] - 투영된 특징
        """
        BV, S, D = resampled_features.shape
        proj_dtype = self._get_module_dtype("vision_to_language_projection", self.vision_to_language_projection, resampled_features.dtype)
        if proj_dtype is not None and resampled_features.dtype != proj_dtype:
            resampled_features = resampled_features.to(proj_dtype)

        # Panorama-aware positional encoding (적용 시점: 리샘플러 출력 직후)
        try:
            if self.use_projection_pe and (self.projection_pe is not None):
                resampled_features = self.projection_pe(resampled_features, batch_size, num_views)
                if proj_dtype is not None and resampled_features.dtype != proj_dtype:
                    resampled_features = resampled_features.to(proj_dtype)
        except Exception:
            # PE 실패 시 원 입력으로 계속 진행 (강건성 우선)
            pass
        
        # Try to infer grid and recombine with overlap-aware stitching after projection
        try:
            H, W = infer_hw(S)
            # [B*V, H, W, D] -> [B, V, H, W, D]
            x5 = resampled_features.view(batch_size, num_views, H, W, D)
            # Project to LM hidden on last dim while keeping 5D shape
            y5 = self.vision_to_language_projection(x5)  # [B, V, H, W, D_lm]
            # Overlap-aware recombination
            y = self._perpare_combining(y5)
            target_dtype = self._get_module_dtype("language_model", self.language_model, y.dtype)
            if target_dtype is not None and y.dtype != target_dtype:
                y = y.to(target_dtype)
            return y  # [B, L_recombined, D_lm]
        except Exception:
            # Fallback: flatten then project then simple concatenation
            x = resampled_features.view(batch_size, num_views * S, D)
            out = self.vision_to_language_projection(x)
            target_dtype = self._get_module_dtype("language_model", self.language_model, out.dtype)
            if target_dtype is not None and out.dtype != target_dtype:
                out = out.to(target_dtype)
            return out  # [B, V*S, D_lm]
    
    def _perpare_combining(self, projected_features: torch.Tensor) -> torch.Tensor:
        """
        오버랩 비율(self.overlap_ratio)에 맞춰 뷰들을 가로로 재구성하여 중복 토큰을 줄입니다.

        stitching_mode:
        - 'drop_overlap': 첫 뷰는 전체, 이후 뷰는 좌측 k=int(W*overlap_ratio) 열을 드랍하여 이어붙임
                         출력 길이 ≈ H * (W + (V-1)*(W-k))
        - 'stride_views': 겹침이 0이 되도록 뷰 인덱스를 s=ceil(1/(1-overlap)) 간격으로 샘플링해 전체 열을 사용
                          (예: overlap=0.5 -> s=2 → 0,2,4,6번째 뷰만 사용)
                          출력 길이 ≈ H * (W * ceil(V/s))
        - 'concat': 단순히 [B,H,V*W,D] 순서로 인터리브 (중복 제거 안 함)

        입력: projected_features [B, V, H, W, D_lm]
        출력: [B, L, D_lm]
        실패 시: 기존 interleave 방식으로 폴백하여 [B, H*V*W, D_lm]
        """
        try:
            y5 = projected_features
            B, V, H, W, D = y5.shape
            ratio = self.overlap_ratio
            k = int(max(0, min(W, round(W * ratio))))

            mode = getattr(self, 'stitching_mode', 'drop_overlap')

            # concat: simple interleave without overlap handling
            if mode == 'concat':
                y = y5.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
                return y.view(B, H * (V * W), D)

            # stride_views: select every s-th view to avoid any overlap entirely
            if mode == 'stride_views':
                # s = ceil(1 / (1 - r)) with protection for r≈1.0
                eps = 1e-8
                denom = max(1.0 - float(ratio), eps)
                s = max(1, int(math.ceil(1.0 / denom)))
                if V <= 1 or s <= 1:
                    # fall back to concat if no need to stride
                    y = y5.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
                    return y.view(B, H * (V * W), D)

                start = int(getattr(self, 'stitch_stride_offset', 0)) % s
                idx = torch.arange(start, V, s, device=y5.device)
                y_sel = y5.index_select(dim=1, index=idx)  # [B, V_sel, H, W, D]
                y = y_sel.permute(0, 2, 1, 3, 4).contiguous().view(B, H, (-1), D)  # [B, H, V_sel*W, D]
                return y.view(B, -1, D)

            # resample: map all views onto a global panorama axis and resample to target width
            if mode == 'resample':
                # compute stride (columns) and unique panorama width in columns
                s_float = (1.0 - float(ratio)) * float(W)
                # union width of V views placed every s columns with each width W:
                # unique_cols = (V-1)*s + W
                unique_cols = max(1, int(round((float(V - 1) * s_float) + float(W))))

                # determine target number of columns
                if int(getattr(self, 'stitch_target_cols', 0)) > 0:
                    T = int(self.stitch_target_cols)
                elif bool(getattr(self, 'stitch_target_to_view_width', False)):
                    T = int(W)
                else:
                    T = unique_cols

                # normalized target x in global panorama axis mapped to each view's local x
                g = torch.linspace(0.0, float(unique_cols) - 1.0, steps=T, device=y5.device)  # [T]
                v_idx = torch.arange(V, device=y5.device, dtype=torch.float32).view(V, 1)     # [V,1]
                x_local = g.view(1, T) - (v_idx * s_float)                                     # [V, T] in [0,W-1]

                # Build sampling grid for grid_sample (differentiable linear sampling)
                # Input for grid_sample: NCHW where N=B*V, C=D, H=H, W=W
                xN = y5.permute(0, 1, 4, 2, 3).contiguous().view(B * V, D, H, W)

                # grid: [N, out_H=H, out_W=T, 2], coords in [-1,1], last dim (x,y)
                # y is identity per row; x is x_local normalized to [-1,1]
                if H > 1:
                    y_lin = torch.linspace(-1.0, 1.0, steps=H, device=y5.device)
                else:
                    y_lin = torch.zeros(1, device=y5.device)
                if W > 1:
                    x_norm = (2.0 * (x_local / float(W - 1))) - 1.0  # [V,T]
                else:
                    x_norm = torch.zeros(V, T, device=y5.device)

                # shape to [V,H,T] and stack to (x,y)
                x_grid = x_norm.view(V, 1, T).expand(V, H, T)
                y_grid = y_lin.view(1, H, 1).expand(V, H, T)
                grid = torch.stack([x_grid, y_grid], dim=-1)  # [V,H,T,2]
                grid = grid.unsqueeze(0).expand(B, V, H, T, 2).contiguous().view(B * V, H, T, 2)

                # choose interpolation mode
                interp = str(getattr(self, 'stitch_interp', 'linear')).lower()
                gs_mode = 'bilinear' if interp == 'linear' else 'nearest'

                # sample values and weights (weights by sampling all-ones)
                sampled = F.grid_sample(xN, grid, mode=gs_mode, padding_mode='zeros', align_corners=True)  # [B*V,D,H,T]
                ones = torch.ones(B * V, 1, H, W, device=y5.device, dtype=y5.dtype)
                w = F.grid_sample(ones, grid, mode=gs_mode, padding_mode='zeros', align_corners=True)      # [B*V,1,H,T]

                # reshape and fuse across views
                sampled = sampled.view(B, V, D, H, T).permute(0, 1, 3, 4, 2)  # [B,V,H,T,D]
                w = w.view(B, V, 1, H, T).permute(0, 1, 3, 4, 2)              # [B,V,H,T,1]
                num = (sampled * w).sum(dim=1)                                 # [B,H,T,D]
                den = w.sum(dim=1).clamp_min(1e-6)                             # [B,H,T,1]
                fused = num / den                                              # [B,H,T,D]
                return fused.reshape(B, -1, D)

            # Default: drop_overlap stitching
            if V <= 1 or k <= 0:
                # No overlap handling needed; interleave views horizontally
                y = y5.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
                return y.view(B, H * (V * W), D)

            # Drop-overlap stitching: first view full, subsequent views drop first k columns
            parts = [y5[:, 0]]  # [B,H,W,D]
            for v in range(1, V):
                parts.append(y5[:, v, :, k:, :])  # [B,H,W-k,D]
            rowwise = torch.cat(parts, dim=2)  # concat along width
            return rowwise.reshape(B, -1, D)
        except Exception:
            # Fallback: attempt to interleave if shape assumptions fail
            try:
                B, V, H, W, D = projected_features.shape
                y = projected_features.permute(0, 2, 1, 3, 4).contiguous().view(B, H, V * W, D)
                return y.view(B, H * (V * W), D)
            except Exception:
                # Last resort: if it's already [B, L, D], return as-is
                return projected_features
    
    def _fuse_text_image_embeddings(self, vision_tokens: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        텍스트 토큰을 임베딩하고 이미지 토큰 자리에 vision tokens 추가
        
        Args:
            vision_tokens: [B, V*S, D_lm] - 투영된 vision tokens
            input_ids: [B, L] - 텍스트 토큰 ID
            
        Returns:
            Dict with inputs_embeds: [B, L'-1+V*S, D_lm], attention_mask, labels
        """
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels)
        return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)
    
    # ==================== 메인 순전파 및 생성 ====================
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "vision",
        **kwargs
    ):
        """
        Args:
            pixel_values: [B,V,C,H,W] 또는 [B,C,H,W]
            stage: "vision" | "resampler" | "finetune"
            **kwargs: 추가 파라미터들 (미래 확장성을 위해 보존)
        """
        # kwargs에서 추가 설정 추출 (필요시)
        debug_mode = kwargs.get('debug_mode', False)
        if debug_mode:
            print(f"[DEBUG] Forward pass - stage: {stage}, batch_size: {pixel_values.size(0)}")
        if stage == "vision":
            # Vision-only stage uses raw vision hidden states (no resampler/projection)
            batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
            vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)
            if num_views <= 1 and not self._warned_single_view:
                print("[VICReg] Warning: num_views <= 1, VICReg 손실이 0이 됩니다.")
                self._warned_single_view = True

            if self.vicreg_loss_weight == 0.0:
                if not hasattr(self, '_warned_zero_vicreg_weight'):
                    print("[VICReg] Warning: VICReg 가중치 0.0 → 계산 스킵")
                    self._warned_zero_vicreg_weight = True
                zero = torch.zeros((), device=vision_hidden_states.device)
                return {"loss": zero, "vicreg_loss": zero, "vicreg_raw": zero, "vicreg_weight": 0.0}

            # NEW: VICReg projector 적용 (vision stage에서만)
            vicreg_feats = self.vicreg_projector(vision_hidden_states)

            vicreg_raw = self._compute_vicreg_overlap_loss(
                vicreg_feats, batch_size, num_views, overlap_ratio=self.overlap_ratio
            )
            vicreg_loss = vicreg_raw * self.vicreg_loss_weight
            total_loss = vicreg_loss
            return {
                "loss": total_loss,
                "vicreg_loss": vicreg_loss,
                "vicreg_raw": vicreg_raw.detach(),
                "vicreg_weight": self.vicreg_loss_weight,
                "vicreg_dim": self._vicreg_feat_dim,
            }

        # vision 외 단계: 공통 처리 파이프라인 사용
        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            
            # 1. Vision encoder 처리
            vision_result = self._process_vision_encoder(pixel_values)
            vision_features = vision_result["vision_features"]  # [B*V, S, D_vision]
            batch_size = vision_result["batch_size"]
            num_views = vision_result["num_views"]
            
            # 2. Resampler 처리
            resampled_features = self._process_resampler(vision_features, batch_size, num_views)  # [B*V, S, D_latent]
            
            # 3. Projection layer 처리
            vision_tokens = self._process_projection_layer(resampled_features, batch_size, num_views)  # [B, V*S, D_lm]
            
            # 4. 텍스트-이미지 융합 및 LM forward
            return self._compute_autoregressive_loss(
                vision_tokens, input_ids, attention_mask, labels
            )

        raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 32, temperature: float = 0.7, **kwargs):
        """forward와 동일한 처리 과정을 사용하는 생성 함수"""
        self.eval()
        try:
            # 1. Vision encoder 처리 (forward와 동일)
            vision_result = self._process_vision_encoder(pixel_values)
            vision_features = vision_result["vision_features"]  # [B*V, S, D_vision]
            batch_size = vision_result["batch_size"]
            num_views = vision_result["num_views"]
            device = vision_result["device"]
            
            # 2. Resampler 처리 (forward와 동일)
            resampled_features = self._process_resampler(vision_features, batch_size, num_views)  # [B*V, S, D_latent]
            
            # 3. Projection layer 처리 (forward와 동일)
            vision_tokens = self._process_projection_layer(resampled_features, batch_size, num_views)  # [B, V*S, D_lm]
            
            # 4. 생성용 입력 준비
            input_ids, attention_mask = self._prepare_generation_inputs(
                input_ids, attention_mask, batch_size, device
            )
            
            # 5. 텍스트-이미지 융합 (forward와 동일, 단 labels 없음)
            combined_inputs = self._fuse_text_image_embeddings(
                vision_tokens, input_ids=input_ids, attention_mask=attention_mask
            )
            
            # 6. LM generate 호출
            generation_kwargs = self._build_lm_generate_kwargs(combined_inputs, max_new_tokens, temperature, **kwargs)
            generated_ids = self.language_model.generate(**generation_kwargs)
            return self._decode_generated_text(generated_ids)
        except Exception as e:
            print(f"[Generate] Error in generation: {e}")
            import traceback; traceback.print_exc()
            return self._fallback_generation_output(pixel_values.size(0) if pixel_values.ndim in (4,5) else 1,
                                                        pixel_values.device if isinstance(pixel_values, torch.Tensor) else 'cpu')

    # ==================== 기본 유틸리티 헬퍼 ====================
    def _normalize_pixel_values(self, pixel_values):
        if pixel_values.ndim == 5:
            batch_size, num_views, _, _, _ = pixel_values.shape
        elif pixel_values.ndim == 4:
            batch_size, _, _, _ = pixel_values.shape
            num_views = 1
            pixel_values = pixel_values.unsqueeze(1)
        else:
            raise ValueError(f"pixel_values shape invalid: {pixel_values.shape}")
        target_dtype = self._get_module_dtype("vision_encoder", self.vision_encoder, pixel_values.dtype)
        if target_dtype is not None and pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(target_dtype)
        return batch_size, num_views, pixel_values

    def _extract_vision_features(self, pixel_values, batch_size, num_views):
        flattened_pixel_values = pixel_values.view(batch_size * num_views, *pixel_values.shape[2:])
        vision_output = self.vision_encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state  # (B*V, S, D)
        vision_hidden_states = self.vicreg_norm(vision_hidden_states)
        return vision_hidden_states

    def _prepare_generation_inputs(self, input_ids, attention_mask, batch_size, device):
        if input_ids is None:
            print("[Generate] Warning: input_ids not provided, creating default prompt for captioning")
            return self._create_default_prompt(batch_size, device)
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            attention_mask = (input_ids != pad_id).long()
        return self._adjust_batch_size(input_ids, attention_mask, batch_size)

    def _create_default_prompt(self, batch_size, device):
        DEFAULT_PROMPT = "<|vision|>\nDescribe this panoramic image in detail."
        DEFAULT_MAX_LENGTH = 64
        encoding = self.tokenizer(DEFAULT_PROMPT, return_tensors="pt",
                                  padding=True, truncation=True, max_length=DEFAULT_MAX_LENGTH)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return self._adjust_batch_size(input_ids, attention_mask, batch_size)

    def _adjust_batch_size(self, input_ids, attention_mask, target_batch_size):
        if input_ids.size(0) == 1 and target_batch_size > 1:
            input_ids = input_ids.expand(target_batch_size, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.expand(target_batch_size, -1)
        return input_ids, attention_mask

    def _get_tokenizer_info(self):
        try:
            if hasattr(self.language_model, 'config'):
                config = self.language_model.config
            elif hasattr(self.language_model, 'base_model') and hasattr(self.language_model.base_model, 'config'):
                config = self.language_model.base_model.config
            else:
                config = None
            if config:
                return {
                    'pad_token_id': getattr(config, 'pad_token_id', None),
                    'eos_token_id': getattr(config, 'eos_token_id', None),
                    'bos_token_id': getattr(config, 'bos_token_id', None),
                }
            return {}
        except Exception:
            return {}

    def _build_generation_kwargs(self, combined_inputs, max_new_tokens, temperature, **kwargs):
        temperature = max(0.1, min(temperature, 1.0))
        MIN_NEW = 5
        DEFAULTS = dict(top_p=0.9, top_k=50, repetition_penalty=1.1, length_penalty=1.0)
        tk = self._get_tokenizer_info()
        out = {
            **combined_inputs,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': kwargs.get('min_new_tokens', MIN_NEW),
            'temperature': temperature,
            'do_sample': temperature > 0.1,
            'top_p': kwargs.get('top_p', DEFAULTS['top_p']),
            'top_k': kwargs.get('top_k', DEFAULTS['top_k']),
            'repetition_penalty': kwargs.get('repetition_penalty', DEFAULTS['repetition_penalty']),
            'length_penalty': kwargs.get('length_penalty', DEFAULTS['length_penalty']),
        }
        if tk.get('pad_token_id') is not None: out['pad_token_id'] = tk['pad_token_id']
        if tk.get('eos_token_id') is not None: out['eos_token_id'] = tk['eos_token_id']
        for key in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
            if key in kwargs:
                out[key] = kwargs[key]
        return out

    def _build_lm_generate_kwargs(self, combined_inputs, max_new_tokens, temperature, **kwargs):
        return self._build_generation_kwargs(combined_inputs, max_new_tokens, temperature, **kwargs)

    def _postprocess_generated_text(self, generated_ids):
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned_texts = [t.strip() for t in generated_texts]
        return {"generated_ids": generated_ids, "text": cleaned_texts}

    def _decode_generated_text(self, generated_ids):
        return self._postprocess_generated_text(generated_ids)

    def _get_fallback_generation_result(self, batch_size, device):
        fallback_text = ["a panoramic view"] * batch_size
        fallback_ids = torch.ones((batch_size, 5), dtype=torch.long, device=device)
        return {"generated_ids": fallback_ids, "text": fallback_text}

    def _fallback_generation_output(self, batch_size, device):
        return self._get_fallback_generation_result(batch_size, device)

    # ==================== VICReg & AR 헬퍼 ====================
    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,   # [B*V, S, D_vicreg]
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,
        pair_chunk: Optional[int] = 8,  # 메모리 절약용 청킹 (None이면 full-batch)
    ):
        """
        벡터화된 VICReg overlap loss 계산
        - 원래 구현과 동일하게 (v, v+1 mod V) 페어별로 VICReg을 구해 평균냄
        - 파이썬 루프 제거 → 큰 폭의 속도 개선
        - 공분산 항은 [P, D, D]를 만들기 때문에 D가 아주 큰 경우 pair_chunk로 나눠 계산 권장

        Args:
            pair_chunk: 한 번에 처리할 페어 수(P=B*V). 메모리 피크 낮추고 싶을 때 설정.
        """
        if num_views <= 1:
            return torch.zeros((), device=vision_output.device)

        # 1) CLS 토큰 제거 및 패치 그리드 복원
        has_cls_token = self._has_cls_token(vision_output)
        patch_features = vision_output[:, 1:] if has_cls_token else vision_output  # [B*V, S', D]
        num_patches = patch_features.size(1)
        grid_h, grid_w = infer_hw(num_patches)
        D = patch_features.size(-1)

        # [B, V, H, W, D]
        patch_features = patch_features.view(batch_size, num_views, grid_h, grid_w, D)

        # 2) 겹치는 칼럼 잘라 (v → 오른쪽), (v+1) → 왼쪽 페어 만들기
        k = max(1, int(grid_w * overlap_ratio))  # 최소 1 칼럼
        curr = patch_features[:, :, :, -k:, :]                    # [B, V, H, k, D]
        nxt  = torch.roll(patch_features, shifts=-1, dims=1)      # (v+1 mod V)
        nxt  = nxt[:, :, :, :k, :]                                # [B, V, H, k, D]

        # 3) [P, L, D]로 평탄화 (P=B*V, L=H*k)
        B, V = batch_size, num_views
        P = B * V
        L = grid_h * k
        curr = curr.contiguous().view(P, L, D)
        nxt  = nxt.contiguous().view(P, L, D)

        # 4) VICReg 항들을 페어별로 한 번에 계산
        # 4-1) invariance (MSE)
        inv_pair = F.mse_loss(curr, nxt, reduction='none').mean(dim=(1, 2))  # [P]

        # 4-2) variance (std를 L축으로 계산 → gamma-margin ReLU → D 평균)
        eps = 1e-4
        gamma = getattr(self.vicreg_loss, 'gamma', 1.0)
        std_x = torch.sqrt(curr.var(dim=1, unbiased=False) + eps)  # [P, D]
        std_y = torch.sqrt(nxt.var(dim=1, unbiased=False) + eps)   # [P, D]
        var_pair = 0.5 * (F.relu(gamma - std_x).mean(dim=1) + F.relu(gamma - std_y).mean(dim=1))  # [P]

        # 4-3) covariance (off-diagonal^2 평균)
        # L축 평균 제거
        curr_c = curr - curr.mean(dim=1, keepdim=True)   # [P, L, D]
        nxt_c  = nxt  - nxt.mean(dim=1,  keepdim=True)   # [P, L, D]
        denom = max(L - 1, 1)

        def _cov_offdiag_sq_mean(xc: torch.Tensor) -> torch.Tensor:
            """
            xc: [P, L, D]
            반환: [P] = sum(offdiag(C)^2)/D,  where C = (xc^T xc)/(L-1)
            메모리 피크를 줄이기 위해 pair_chunk로 나눠서 처리 가능
            """
            P = xc.size(0)
            if pair_chunk is None or pair_chunk >= P:
                # full-batch
                C = torch.bmm(xc.transpose(1, 2), xc) / denom   # [P, D, D]
                C2_sum = (C ** 2).sum(dim=(1, 2))               # ||C||_F^2
                diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                offdiag_sq = C2_sum - diag_sq                   # sum of off-diagonal^2
                return offdiag_sq / D
            else:
                # chunked accumulation
                out = []
                for s in range(0, P, pair_chunk):
                    e = min(s + pair_chunk, P)
                    C = torch.bmm(xc[s:e].transpose(1, 2), xc[s:e]) / denom  # [p, D, D]
                    C2_sum = (C ** 2).sum(dim=(1, 2))
                    diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                    offdiag_sq = C2_sum - diag_sq
                    out.append(offdiag_sq / D)
                    del C  # 메모리 즉시 해제
                return torch.cat(out, dim=0)

        cov_x = _cov_offdiag_sq_mean(curr_c)  # [P]
        cov_y = _cov_offdiag_sq_mean(nxt_c)   # [P]
        cov_pair = 0.5 * (cov_x + cov_y)      # [P]

        # 5) 가중합 → 페어 평균 → 클램프
        w_inv = getattr(self.vicreg_loss, 'similarity_weight', 25.0)
        w_var = getattr(self.vicreg_loss, 'variance_weight', 25.0)
        w_cov = getattr(self.vicreg_loss, 'covariance_weight', 1.0)

        per_pair = w_inv * inv_pair + w_var * var_pair + w_cov * cov_pair  # [P]
        total = per_pair.mean()
        return torch.clamp(total, max=1e6)


    def _prepare_text_inputs(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        if labels is None:
            labels = input_ids.clone()
        max_len = min(input_ids.size(1), self.max_text_length)
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        labels = labels[:, :max_len]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'batch_size': batch_size}

    def _get_module_dtype(self, cache_key: str, module: nn.Module, default: torch.dtype | None = None) -> torch.dtype | None:
        if cache_key in self._cached_module_dtypes:
            return self._cached_module_dtypes[cache_key]
        dtype = None
        try:
            param = next(module.parameters())
            dtype = param.dtype
        except StopIteration:
            pass
        except AttributeError:
            dtype = None
        if dtype is None:
            try:
                buf = next(module.buffers())
                dtype = buf.dtype
            except (StopIteration, AttributeError):
                dtype = None
        if dtype is None:
            dtype = default
        if dtype is not None:
            self._cached_module_dtypes[cache_key] = dtype
        return dtype

    def _create_combined_inputs(self, vision_tokens, input_ids=None, attention_mask=None, labels=None, text_inputs=None):
        """
        학습 시 결합 로직: 텍스트 내 '<|vision|>' 토큰 위치에 비전 토큰 시퀀스를 삽입.
        placeholder가 없으면 이전 방식대로 비전 토큰을 프리픽스.
        라벨은 삽입된 비전 토큰 구간을 ignore_index로 채움.
        """
        device = vision_tokens.device
        if text_inputs is not None:
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            labels = text_inputs.get('labels')
            batch_size = text_inputs['batch_size']
        else:
            batch_size = input_ids.size(0)

        # 텍스트 임베딩
        if hasattr(self.language_model, 'get_input_embeddings'):
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        elif hasattr(self.language_model, 'base_model'):
            text_embeddings = self.language_model.base_model.get_input_embeddings()(input_ids)
        else:
            text_embeddings = self.language_model.model.embed_tokens(input_ids)

        # Ensure vision tokens collapse to [batch, seq, hidden] even when stitching
        # keeps extra spatial dims or when a single sample squeezes the batch axis.
        if vision_tokens.dim() == 2:
            vision_tokens = vision_tokens.unsqueeze(0)
        if vision_tokens.dim() > 3:
            vision_tokens = vision_tokens.view(batch_size, -1, vision_tokens.size(-1))
        elif vision_tokens.dim() == 3 and vision_tokens.size(0) != batch_size:
            seq = max(1, vision_tokens.numel() // (batch_size * vision_tokens.size(-1)))
            vision_tokens = vision_tokens.reshape(batch_size, seq, vision_tokens.size(-1))
        vision_tokens = vision_tokens.contiguous()

        # 크기 정합성
        if vision_tokens.size(0) != batch_size:
            min_batch = min(vision_tokens.size(0), batch_size)
            vision_tokens = vision_tokens[:min_batch]
            text_embeddings = text_embeddings[:min_batch]
            attention_mask = attention_mask[:min_batch]
            if labels is not None:
                labels = labels[:min_batch]
            batch_size = min_batch

        bsz, text_len, hidden = text_embeddings.shape
        vis_len = vision_tokens.size(1)
        vt_id = getattr(self, 'vision_token_id', None)

        out_embeds = []
        out_attn = []
        out_labels = [] if labels is not None else None

        for b in range(bsz):
            ids_b = input_ids[b]
            emb_b = text_embeddings[b]
            attn_b = attention_mask[b]
            lbl_b = labels[b] if labels is not None else None

            # vision 토큰 위치
            if vt_id is not None and vt_id >= 0:
                pos_list = (ids_b == vt_id).nonzero(as_tuple=False).flatten().tolist()
            else:
                pos_list = []

            if len(pos_list) == 0:
                # 폴백: 비전 프리픽스
                new_emb = torch.cat([vision_tokens[b], emb_b], dim=0)
                new_attn = torch.cat([
                    torch.ones(vis_len, dtype=attn_b.dtype, device=device),
                    attn_b
                ], dim=0)
                out_embeds.append(new_emb)
                out_attn.append(new_attn)
                if out_labels is not None:
                    ignore = torch.full((vis_len,), self.ignore_index, dtype=lbl_b.dtype, device=device)
                    new_lbl = torch.cat([ignore, lbl_b], dim=0)
                    out_labels.append(new_lbl)
                continue

            cur_emb = []
            cur_attn = []
            cur_lbl = [] if out_labels is not None else None
            last = 0
            for pos in pos_list:
                # 앞쪽 텍스트 조각
                if pos > last:
                    cur_emb.append(emb_b[last:pos])
                    cur_attn.append(attn_b[last:pos])
                    if cur_lbl is not None:
                        cur_lbl.append(lbl_b[last:pos])
                # 비전 시퀀스 삽입 (라벨은 ignore)
                cur_emb.append(vision_tokens[b])
                cur_attn.append(torch.ones(vis_len, dtype=attn_b.dtype, device=device))
                if cur_lbl is not None:
                    cur_lbl.append(torch.full((vis_len,), self.ignore_index, dtype=lbl_b.dtype, device=device))
                last = pos + 1
            # 꼬리 텍스트
            if last < text_len:
                cur_emb.append(emb_b[last:])
                cur_attn.append(attn_b[last:])
                if cur_lbl is not None:
                    cur_lbl.append(lbl_b[last:])

            new_emb = torch.cat(cur_emb, dim=0)
            new_attn = torch.cat(cur_attn, dim=0)
            out_embeds.append(new_emb)
            out_attn.append(new_attn)
            if out_labels is not None:
                out_labels.append(torch.cat(cur_lbl, dim=0))

        # 배치 패딩
        max_len = max(e.size(0) for e in out_embeds)
        padded_embeds = torch.zeros(bsz, max_len, hidden, device=device, dtype=text_embeddings.dtype)
        padded_attn = torch.zeros(bsz, max_len, device=device, dtype=attention_mask.dtype)
        if out_labels is not None:
            # dtype은 labels의 dtype을 따름
            padded_labels = torch.full((bsz, max_len), self.ignore_index, device=device, dtype=labels.dtype)
        else:
            padded_labels = None
        for b in range(bsz):
            L = out_embeds[b].size(0)
            padded_embeds[b, :L] = out_embeds[b]
            padded_attn[b, :L] = out_attn[b]
            if padded_labels is not None:
                padded_labels[b, :L] = out_labels[b]

        result = {'inputs_embeds': padded_embeds, 'attention_mask': padded_attn}
        if padded_labels is not None:
            result['labels'] = padded_labels
        return result

    def _compose_multimodal_embeddings(self, vision_tokens, input_ids=None, attention_mask=None, labels=None, text_inputs=None):
        return self._create_combined_inputs(vision_tokens, input_ids=input_ids, attention_mask=attention_mask, labels=labels, text_inputs=text_inputs)

    # Gradient checkpointing utilities -------------------------------------------------
    def gradient_checkpointing_enable(self, use_reentrant: bool | None = None):
        """Enable gradient checkpointing for submodules that support it."""
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            try:
                kwargs = {}
                if use_reentrant is not None:
                    kwargs["use_reentrant"] = use_reentrant
                self.language_model.gradient_checkpointing_enable(**kwargs)
            except TypeError:
                self.language_model.gradient_checkpointing_enable()
        if hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
            try:
                self.vision_encoder.gradient_checkpointing_enable()
            except TypeError:
                self.vision_encoder.gradient_checkpointing_enable()
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.language_model, "gradient_checkpointing_disable"):
            self.language_model.gradient_checkpointing_disable()
        if hasattr(self.vision_encoder, "gradient_checkpointing_disable"):
            self.vision_encoder.gradient_checkpointing_disable()
        self._gradient_checkpointing = False

    def _compute_autoregressive_loss(self, vision_tokens, input_ids, attention_mask, labels):
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels)
        combined_inputs = self._create_combined_inputs_for_training(vision_tokens, text_inputs)
        try:
            outputs = self.language_model(
                inputs_embeds=combined_inputs['inputs_embeds'],
                attention_mask=combined_inputs['attention_mask'],
                labels=combined_inputs['labels'],
                return_dict=True
            )
            if not torch.isfinite(outputs.loss):
                manual = self._compute_manual_cross_entropy_loss(outputs.logits, combined_inputs['labels'])
                manual.setdefault('ar_loss', manual['loss'])
                return manual
            if hasattr(self, '_debug_loss_verification') and self._debug_loss_verification:
                manual = self._compute_manual_cross_entropy_loss(outputs.logits, combined_inputs['labels'])
                diff = abs(outputs.loss.item() - manual['loss'].item())
                if diff > 0.01:
                    print(f"[Loss Debug] HF loss: {outputs.loss.item():.6f}, Manual loss: {manual['loss'].item():.6f}, Diff: {diff:.6f}")
            return {"loss": outputs.loss, "ar_loss": outputs.loss, "logits": outputs.logits}
        except Exception:
            fallback_loss = torch.tensor(0.0, device=vision_tokens.device, requires_grad=True)
            return {"loss": fallback_loss, "ar_loss": fallback_loss,
                    "logits": torch.zeros((text_inputs['batch_size'], combined_inputs['inputs_embeds'].size(1),
                                           self.language_model.config.vocab_size), device=vision_tokens.device)}

    def _project_to_lm_hidden(self, image_features):
        return self.vision_to_language_projection(image_features)

    def _create_combined_inputs_for_training(self, vision_tokens, text_inputs):
        return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)

    def _compute_manual_cross_entropy_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not hasattr(self, '_debug_shift_logged'):
            valid_tokens = (shift_labels != self.ignore_index).sum()
            total_tokens = shift_labels.numel()
            print(f"[Loss Debug] logits: {logits.shape} -> {shift_logits.shape}, labels: {labels.shape} -> {shift_labels.shape}")
            print(f"[Loss Debug] Valid tokens: {valid_tokens.item()}/{total_tokens} ({valid_tokens.item()/total_tokens*100:.1f}%)")
            self._debug_shift_logged = True
        return {"loss": loss, "ar_loss": loss, "perplexity": torch.exp(loss) if torch.isfinite(loss) else torch.tensor(float('inf'))}


    # ==================== LoRA 유틸 (생략 없이 유지) ====================
    def setup_lora_for_finetune(self, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1, target_modules: Optional[list] = None) -> bool:
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. LoRA setup skipped.")
            return False
        try:
            if target_modules is None:
                target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
            lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                                     lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
            self.language_model = get_peft_model(self.language_model, lora_config)
            if hasattr(self.language_model, 'enable_input_require_grads'):
                self.language_model.enable_input_require_grads()
            trainable_params = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.language_model.parameters())
            print(f"✓ LoRA setup: trainable {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
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
                print("Model already PEFT-enabled. Trying to load adapter...")
                if hasattr(self.language_model, 'load_adapter'):
                    adapter_name = "eval_adapter"
                    try:
                        self.language_model.load_adapter(load_path, adapter_name=adapter_name)
                        self.language_model.set_adapter(adapter_name)
                        print(f"✓ LoRA adapter '{adapter_name}' loaded")
                        return True
                    except Exception as e:
                        print(f"load_adapter failed, fallback to manual state_dict load: {e}")
                import os
                state_dict = None
                # safetensors 우선 -> 실패 시 PyTorch bin 시도
                st_path = os.path.join(load_path, "adapter_model.safetensors")
                if os.path.exists(st_path):
                    try:
                        from safetensors.torch import load_file as load_sft
                        state_dict = load_sft(st_path)
                    except Exception as _e:
                        print(f"Warning: failed to load safetensors adapter: {_e}")
                        state_dict = None
                if state_dict is None:
                    bin_path = os.path.join(load_path, "adapter_model.bin")
                    if os.path.exists(bin_path):
                        try:
                            import torch
                            state_dict = torch.load(bin_path, map_location='cpu')
                        except Exception as _e:
                            print(f"Warning: failed to load bin adapter: {_e}")
                            state_dict = None
                if state_dict is None:
                    print(f"Error: No adapter weights found in {load_path} (adapter_model.safetensors or adapter_model.bin)")
                    return False
                cleaned = {}
                for k, v in state_dict.items():
                    ck = k
                    for prefix in ('base_model.model.', 'base_model.'):
                        if ck.startswith(prefix):
                            ck = ck[len(prefix):]
                            break
                    cleaned[ck] = v
                missing, unexpected = self.language_model.load_state_dict(cleaned, strict=False)
                loaded_cnt = len(cleaned) - len(missing)
                if loaded_cnt == 0:
                    print("Error: No LoRA weights matched.")
                    return False
                print(f"✓ LoRA weights loaded: {loaded_cnt}/{len(cleaned)} (missing {len(missing)}, unexpected {len(unexpected)})")
                return True
            else:
                print("Converting to PeftModel.from_pretrained...")
                self.language_model = PeftModel.from_pretrained(self.language_model, load_path, is_trainable=False)
                print("✓ LoRA weights loaded via PeftModel")
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
                adapter_name = list(peft_config.keys())[0]
                config = peft_config.get(adapter_name, None)
                if config is not None:
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
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot save LoRA weights.")
            return False
        try:
            from pathlib import Path
            lora_info = self.get_lora_info()
            if not lora_info.get("is_lora_enabled", False):
                print("Warning: LoRA is not enabled. No weights to save.")
                return False
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            # safetensors를 사용하지 않도록 safe_serialization=False로 강제
            if safe_save_pretrained(self.language_model, str(save_dir), safe_serialization=False):
                print(f"✓ LoRA weights saved: {save_dir}")
                return True
            else:
                print("Error: Failed to save LoRA weights")
                return False
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")
            import traceback; traceback.print_exc()
            return False

    # (저장 메서드 삭제: 체크포인트 저장은 Lightning의 ModelCheckpoint에 위임)

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
            checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"체크포인트 로딩 실패: {e}")
        
        # 하이퍼파라미터 추출
        hparams = checkpoint.get('hyper_parameters', {})
        model_state_dict = checkpoint.get('state_dict', {})

        # 체크포인트로부터 투영 레이어 출력 차원 추정 (LM hidden size 유도)
        proj_out_dim = None
        try:
            for k in [
                'model.vision_to_language_projection.weight',
                'vision_to_language_projection.weight'
            ]:
                if k in model_state_dict:
                    w = model_state_dict[k]
                    # torch.Size([out_features, in_features])
                    if hasattr(w, 'shape') and len(w.shape) == 2:
                        proj_out_dim = int(w.shape[0])
                        break
        except Exception:
            proj_out_dim = None

        # Qwen2.5 계열의 hidden_size 추정 → 모델 이름 추론 (필요 시만 사용)
        def _infer_qwen_from_hidden_size(hs: Optional[int]) -> Optional[str]:
            if hs is None:
                return None
            # 최소한의 매핑만 제공 (현재 코드베이스에서 사용하는 모델들)
            mapping = {
                896:  'Qwen/Qwen2.5-0.5B-Instruct',
                1536: 'Qwen/Qwen2.5-1.5B-Instruct',
            }
            return mapping.get(int(hs))
        
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
                    'overlap_ratio': hparams.get('overlap_ratio', 0.5),
                    'max_text_length': hparams.get('max_text_length', 512),
                }
                model_config = ModelConfig.from_dict(config_dict)
            
            # 2.5 체크포인트 하이퍼파라미터를 우선 적용하여 구조적 불일치 최소화
            #     (auto_detect_config가 전역 config.json을 잡을 수 있으므로, hparams로 핵심 값 동기화)
            if isinstance(model_config, ModelConfig) and isinstance(hparams, dict) and len(hparams) > 0:
                override_keys = [
                    'vision_name', 'language_model_name', 'resampler_type',
                    'latent_dimension', 'max_text_length', 'vicreg_loss_weight', 'overlap_ratio',
                    'resampler_depth', 'resampler_hidden_dim', 'resampler_use_ln',
                    'resampler_num_latents', 'resampler_heads', 'resampler_dropout'
                ]
                hp_overrides = {k: v for k, v in hparams.items() if k in override_keys}
                if hp_overrides:
                    try:
                        model_config = model_config.update(**hp_overrides)
                        print(f"🧩 체크포인트 하이퍼파라미터로 핵심 설정 동기화: {list(hp_overrides.keys())[:5]}{'...' if len(hp_overrides)>5 else ''}")
                    except Exception as _e:
                        print(f"[from_checkpoint] Warning: failed to merge hparams into config: {_e}")

            # 투영 레이어로부터 유도된 LM 이름을 우선 고려 (체크포인트와 구조 일치 보장)
            inferred_lm_name = _infer_qwen_from_hidden_size(proj_out_dim)
            if inferred_lm_name:
                try:
                    model_config = model_config.update(language_model_name=inferred_lm_name)
                    print(f"🧭 체크포인트 투영 차원({proj_out_dim})에 맞춰 LM 고정: {inferred_lm_name}")
                except Exception as _e:
                    print(f"[from_checkpoint] Warning: failed to enforce LM from projection dim: {_e}")

            # 3. 사용자 지정 파라미터로 오버라이드 (허용된 키만)
            if model_kwargs:
                try:
                    allowed = set(ModelConfig().__dict__.keys())
                except Exception:
                    allowed = set()
                filtered = {k: v for k, v in model_kwargs.items() if k in allowed}
                # 만약 체크포인트에서 LM 차원을 유도했으면, 사용자 입력으로 인한 구조 불일치 방지
                if inferred_lm_name and 'language_model_name' in filtered and filtered['language_model_name'] != inferred_lm_name:
                    print(f"⚠️  사용자 LM({filtered['language_model_name']})가 체크포인트 투영 차원({proj_out_dim})과 불일치 → 무시하고 {inferred_lm_name} 사용")
                    filtered.pop('language_model_name', None)
                if filtered:
                    print(f"🛠️  사용자 파라미터로 설정 오버라이드: {list(filtered.keys())}")
                    try:
                        model_config = model_config.update(**filtered)
                    except Exception as _e:
                        print(f"[from_checkpoint] Warning: failed to apply user kwargs: {_e}")
            
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
                'overlap_ratio': 0.5,
                'max_text_length': 512,
            }
            
            # 하이퍼파라미터에서 업데이트
            for key in model_params.keys():
                if key in hparams:
                    model_params[key] = hparams[key]
            
            # 사용자 지정 파라미터로 최종 업데이트
            model_params.update(model_kwargs)
        
        print(f"🛠️  모델 파라미터:")
        preview = {k: v for k, v in model_params.items() if k != 'config'}
        for i, (key, value) in enumerate(preview.items()):
            if i >= 12:
                print("   - ...")
                break
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
            loaded_cnt = len(model_weights) - len(missing_keys)
            print(f"   - 로드된 키: {loaded_cnt}")
            if missing_keys:
                print(f"   - 누락된 키: {len(missing_keys)} (예: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''})")
            if unexpected_keys:
                print(f"   - 예상치 못한 키: {len(unexpected_keys)} (예: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''})")
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
                tokenizer_name = model_params.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
                model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"   ✅ 토크나이저 로드: {tokenizer_name}")
            except Exception as e:
                print(f"   ⚠️ 토크나이저 로드 실패: {e}")
        
        print(f"✅ 모델 로딩 완료 - Device: {device}")
        return model

    @classmethod
    def from_pretrained_dir(
        cls,
        pretrained_dir: str,
        device: str = "auto",
        strict_loading: bool = False,
        **model_kwargs
    ) -> 'PanoramaVLM':
        """
        HuggingFace-style로 저장된 디렉토리에서 모델 로드.
        - save_pretrained로 저장된 폴더 구조를 기대 (pytorch_model.bin, config.json)
        - config.json이 있으면 파라미터 추출, 전달된 model_kwargs가 우선 적용
        """
        from pathlib import Path
        import json

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)

        p = Path(pretrained_dir)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Pretrained directory not found: {pretrained_dir}")

        # 기본 파라미터
        params = {
            'vision_name': 'google/siglip-base-patch16-224',
            'language_model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
            'resampler_type': 'mlp',
            'latent_dimension': 768,
            'vicreg_loss_weight': 1.0,
            'overlap_ratio': 0.5,
            'max_text_length': 512,
        }

        # config.json에서 보정
        cfg_path = p / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    saved_cfg = json.load(f)
                params.update({
                    'vision_name': saved_cfg.get('vision_name', params['vision_name']),
                    'language_model_name': saved_cfg.get('language_model_name', params['language_model_name']),
                    'latent_dimension': saved_cfg.get('latent_dimension', params['latent_dimension']),
                    'max_text_length': saved_cfg.get('max_text_length', params['max_text_length']),
                    'vicreg_loss_weight': saved_cfg.get('vicreg_loss_weight', params['vicreg_loss_weight']),
                    'overlap_ratio': saved_cfg.get('overlap_ratio', params['overlap_ratio']),
                })
            except Exception as e:
                print(f"[from_pretrained_dir] Warning: failed to parse config.json: {e}")

        # 외부 전달 인자 우선하되 'config'/'model_config' 키는 무시하여 저장된 설정과의 구조 불일치를 방지
        if model_kwargs:
            filtered = {k: v for k, v in model_kwargs.items() if k not in ("config", "model_config")}
            if filtered:
                params.update(filtered)

        print(f"🏗️  모델 인스턴스 생성(from_pretrained_dir): {pretrained_dir}")
        model = cls(**params)

        # 가중치 파일 찾기 (PyTorch bin만 지원)
        state_path = None
        if (p/"pytorch_model.bin").exists():
            state_path = p/"pytorch_model.bin"
            try:
                state = torch.load(str(state_path), map_location='cpu')
            except Exception as e:
                print(f"[from_pretrained_dir] Failed to load torch model: {e}")
                state = None
        else:
            state = None

        if state is not None:
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state, strict=strict_loading)
                print(f"   ✅ Weights loaded from {state_path}")
                if missing_keys:
                    print(f"   ⚠️ Missing keys: {len(missing_keys)} (showing first 5) {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"   ⚠️ Unexpected keys: {len(unexpected_keys)} (showing first 5) {unexpected_keys[:5]}")
            except Exception as e:
                print(f"   ❌ Failed to load state dict: {e}")
        else:
            print(f"   ⚠️ No state dict found in {pretrained_dir}. Using randomly initialized weights.")

        model = model.to(device_obj)
        model.eval()
        # 토크나이저 보장
        if not hasattr(model, 'tokenizer'):
            try:
                model.tokenizer = AutoTokenizer.from_pretrained(params.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'))
            except Exception:
                pass
        print(f"✅ 모델 로딩 완료 - Device: {device}")
        return model

# -------------------- 편의: combined inputs 생성 (공용) --------------------
# def _stack_pad_mask(mask_list):  # 사용되지 않음
#     return torch.cat(mask_list, dim=1)

# 클래스 메서드에 두기 애매한 보조 함수들 보완
def _create_combined_inputs_for_training(self, vision_tokens, text_inputs):
    return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)

# 바인딩
PanoramaVLM._create_combined_inputs_for_training = _create_combined_inputs_for_training
