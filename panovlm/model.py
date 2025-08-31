# coding: utf-8

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .utils import set_seed, safe_save_pretrained, infer_hw
from .losses import VicRegLoss
from .resampler.resamplers import MLPResampler

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
    파노라마 전용 Positional Encoding
    
    특징:
    1. View-aware positioning: 각 뷰의 각도 정보 인코딩
    2. Spatial positioning: 각 패치의 공간적 위치 인코딩
    3. Cross-view continuity: 인접 뷰 간 연속성 보장
    4. Multi-scale encoding: 다양한 스케일의 위치 정보 통합
    """
    def __init__(
        self,
        embed_dim: int,
        num_views: int = 8,
        max_patches: int = 256,
        view_encoding_type: str = "sinusoidal",  # "sinusoidal", "learned", "mixed"
        spatial_encoding_type: str = "sinusoidal",  # "sinusoidal", "learned", "mixed"
        enable_continuity: bool = True,
        temperature: float = 10000.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        self.max_patches = max_patches
        self.view_encoding_type = view_encoding_type
        self.spatial_encoding_type = spatial_encoding_type
        self.enable_continuity = enable_continuity
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # View angle encoding (각 뷰의 방위각 정보)
        if view_encoding_type == "learned":
            self.view_embedding = nn.Embedding(num_views, embed_dim)
        elif view_encoding_type == "mixed":
            self.view_embedding = nn.Embedding(num_views, embed_dim // 2)
            
        # Spatial position encoding (패치 레벨 공간 위치)
        if spatial_encoding_type == "learned":
            self.spatial_embedding = nn.Embedding(max_patches, embed_dim)
        elif spatial_encoding_type == "mixed":
            self.spatial_embedding = nn.Embedding(max_patches, embed_dim // 2)
            
        # Cross-view continuity enhancement
        if enable_continuity:
            self.continuity_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.Sigmoid()
            )
            
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def _get_sinusoidal_encoding(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal positional encoding"""
        batch_size, seq_len = positions.shape
        device = positions.device
        
        # Create dimension indices
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * 
                           -(math.log(self.temperature) / dim))
        
        # Expand positions for broadcasting
        pos_expanded = positions.float().unsqueeze(-1)  # [B, L, 1]
        
        # Calculate sin and cos
        pe = torch.zeros(batch_size, seq_len, dim, device=device)
        pe[:, :, 0::2] = torch.sin(pos_expanded * div_term)
        pe[:, :, 1::2] = torch.cos(pos_expanded * div_term)
        
        return pe
    
    def _get_spatial_positions(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generate spatial position indices for patches"""
        # Convert 2D coordinates to 1D position index for encoding
        spatial_indices = torch.arange(height * width, device=device)
        return spatial_indices
    
    def forward(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Apply panorama-aware positional encoding
        
        Args:
            features: [B*V, S, D] - Input features
            batch_size: B
            num_views: V
            
        Returns:
            encoded_features: [B*V, S, D] - Position-encoded features
        """
        BV, S, D = features.shape
        device = features.device
        
        # Infer spatial dimensions
        grid_h, grid_w = infer_hw(S)
        
        # 1. View-level encoding
        # Create view indices for each sample in the flattened format [B*V]
        # For example, if B=2, V=8: [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7]
        view_ids = torch.arange(num_views, device=device).repeat(batch_size)  # [B*V]
        
        if self.view_encoding_type == "sinusoidal":
            # Convert view IDs to normalized positions [0, 1)
            view_positions = view_ids.float() / num_views  # [B*V]
            view_pe = self._get_sinusoidal_encoding(
                view_positions.unsqueeze(1).expand(-1, S), D
            )  # [B*V, S, D]
        elif self.view_encoding_type == "learned":
            view_pe = self.view_embedding(view_ids).unsqueeze(1).expand(-1, S, -1)  # [B*V, S, D]
        elif self.view_encoding_type == "mixed":
            # Half learned, half sinusoidal
            learned_pe = self.view_embedding(view_ids).unsqueeze(1).expand(-1, S, -1)  # [B*V, S, D//2]
            view_positions = view_ids.float() / num_views  # [B*V]
            sin_pe = self._get_sinusoidal_encoding(
                view_positions.unsqueeze(1).expand(-1, S), D // 2
            )  # [B*V, S, D//2]
            view_pe = torch.cat([learned_pe, sin_pe], dim=-1)  # [B*V, S, D]
        else:
            view_pe = torch.zeros_like(features)
            
        # 2. Spatial-level encoding
        spatial_indices = self._get_spatial_positions(grid_h, grid_w, device)
        spatial_indices = spatial_indices.unsqueeze(0).expand(BV, -1)  # [B*V, S]
        
        if self.spatial_encoding_type == "sinusoidal":
            spatial_pe = self._get_sinusoidal_encoding(spatial_indices, D)  # [B*V, S, D]
        elif self.spatial_encoding_type == "learned":
            spatial_pe = self.spatial_embedding(spatial_indices)  # [B*V, S, D]
        elif self.spatial_encoding_type == "mixed":
            learned_pe = self.spatial_embedding(spatial_indices)  # [B*V, S, D//2]
            sin_pe = self._get_sinusoidal_encoding(spatial_indices, D // 2)  # [B*V, S, D//2]
            spatial_pe = torch.cat([learned_pe, sin_pe], dim=-1)  # [B*V, S, D]
        else:
            spatial_pe = torch.zeros_like(features)
            
        # 3. Combine encodings
        combined_pe = view_pe + spatial_pe
        
        # 4. Cross-view continuity enhancement
        if self.enable_continuity:
            continuity_weights = self.continuity_mlp(combined_pe)
            combined_pe = combined_pe * continuity_weights
            
        # 5. Apply to features
        encoded_features = features + combined_pe
        encoded_features = self.norm(encoded_features)
        encoded_features = self.dropout(encoded_features)
        
        return encoded_features


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

        # 언어 모델 및 투영 ------------------------------------
        lm_name = getattr(self.config, 'language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
        self.language_model = AutoModelForCausalLM.from_pretrained(lm_name,
                                                                   attn_implementation="sdpa",
                                                                   )
        self.vision_to_language_projection = nn.Linear(latent_dimension, self.language_model.config.hidden_size)

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
        self.vicreg_overlap_ratio = float(getattr(self.config, 'vicreg_overlap_ratio', 0.5))

        # 텍스트 설정 ------------------------------------------
        self.max_text_length = int(getattr(self.config, 'max_text_length', 512))
        self.ignore_index = -100

        # 디버그 플래그
        self._warned_single_view = False
        self._debug_loss_verification = False

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
        
        # Try to infer grid and interleave views horizontally for panorama continuity
        try:
            H, W = infer_hw(S)
            # [B*V, H, W, D] -> [B, V, H, W, D]
            x = resampled_features.view(batch_size, num_views, H, W, D)
            # [B, V, H, W, D] -> [B, H, V, W, D] (view interleaving)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # [B, H, V, W, D] -> [B, H, V*W, D]
            x = x.view(batch_size, H, num_views * W, D)
            # [B, H, V*W, D] -> [B, V*S, D]
            x = x.reshape(batch_size, H * (num_views * W), D)
        except Exception:
            # Fallback: simple view concatenation
            x = resampled_features.view(batch_size, num_views * S, D)
        
        # Project to LM hidden dimension
        return self.vision_to_language_projection(x)  # [B, V*S, D_lm]
    
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
                vicreg_feats, batch_size, num_views, overlap_ratio=self.vicreg_overlap_ratio
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
                from safetensors.torch import load_file
                adapter_weights_path = os.path.join(load_path, "adapter_model.safetensors")
                if not os.path.exists(adapter_weights_path):
                    print(f"Error: adapter_model.safetensors not found in {load_path}")
                    return False
                state_dict = load_file(adapter_weights_path)
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
            if safe_save_pretrained(self.language_model, str(save_dir), safe_serialization=True):
                print(f"✓ LoRA weights saved: {save_dir}")
                return True
            else:
                print("Error: Failed to save LoRA weights")
                return False
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")
            import traceback; traceback.print_exc()
            return False

    # ==================== Save/Load (요약) ====================
    def save_pretrained(self, save_directory: str, save_lora_separately: bool = True):
        from pathlib import Path
        import json
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 모델 저장 중: {save_directory}")
        try:
            from safetensors.torch import save_file
            model_path = save_dir / "model.safetensors"
            save_file(self.state_dict(), model_path)
            print(f"   ✅ 모델 가중치 저장 (SafeTensors): {model_path}")
        except ImportError:
            model_path = save_dir / "pytorch_model.bin"
            torch.save(self.state_dict(), model_path)
            print(f"   ✅ 모델 가중치 저장 (PyTorch): {model_path}")

        # 간단한 config json 저장
        config = {
            "model_type": "PanoramaVLM",
            "vision_name": getattr(self.vision_encoder.config, 'name_or_path', getattr(self.config, 'vision_name', 'unknown')),
            "language_model_name": getattr(self.language_model.config, 'name_or_path', getattr(self.config, 'language_model_name', 'unknown')),
            "latent_dimension": self.vision_to_language_projection.in_features,
            "max_text_length": self.max_text_length,
            "vicreg_loss_weight": self.vicreg_loss_weight,
            "vicreg_overlap_ratio": self.vicreg_overlap_ratio,
            "use_vicreg_projector": self.use_vicreg_projector,
            "vicreg_projector_dim": self.vicreg_projector_dim,
            "vicreg_projector_depth": self.vicreg_projector_depth,
            "vicreg_projector_hidden": self.vicreg_projector_hidden,
            "vicreg_projector_ln": self.vicreg_projector_ln,
            "use_vicreg_norm": self.use_vicreg_norm,
        }
        with open(save_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        if save_lora_separately:
            lora_dir = save_dir / "lora_weights"
            _ = self.save_lora_weights(str(lora_dir))

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
            if key != 'config':  # config 객체는 출력하지 않음
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
        - save_pretrained로 저장된 폴더 구조를 기대 (model.safetensors 또는 pytorch_model.bin, config.json)
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
            'vicreg_overlap_ratio': 0.5,
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
                    'vicreg_overlap_ratio': saved_cfg.get('vicreg_overlap_ratio', params['vicreg_overlap_ratio']),
                })
            except Exception as e:
                print(f"[from_pretrained_dir] Warning: failed to parse config.json: {e}")

        # 외부 전달 인자 우선
        params.update(model_kwargs or {})

        print(f"🏗️  모델 인스턴스 생성(from_pretrained_dir): {pretrained_dir}")
        model = cls(**params)

        # 가중치 파일 찾기
        state_path = None
        if (p/"model.safetensors").exists():
            state_path = p/"model.safetensors"
            try:
                from safetensors.torch import load_file as load_safetensors
                state = load_safetensors(str(state_path))
            except Exception as e:
                print(f"[from_pretrained_dir] Failed to load safetensors: {e}")
                state = None
        elif (p/"pytorch_model.bin").exists():
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
