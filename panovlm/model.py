# coding: utf-8

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# LoRA 지원을 위한 PEFT import (선택적)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA fine-tuning will not be supported.")

# ==================== Reproducibility Utility ====================
import os, random, numpy as np

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        else:
            torch.backends.cudnn.benchmark = True
    except Exception as _e:
        print(f"[set_seed] Torch seed setup skipped: {_e}")

def _safe_save_pretrained(model, save_path: str, **kwargs) -> bool:
    if not hasattr(model, 'save_pretrained'):
        return False
    safe_kwargs = {
        'push_to_hub': False,
        'token': False,
        'safe_serialization': kwargs.get('safe_serialization', True),
        **kwargs
    }
    for param in ['repo_id', 'from_id', 'to_id', 'hub_model_id']:
        safe_kwargs.pop(param, None)
    try:
        model.save_pretrained(save_path, **safe_kwargs)
        return True
    except Exception as e:
        print(f"Warning: Failed to save with SafeTensors: {e}")
        try:
            fallback_kwargs = {k: v for k, v in safe_kwargs.items() if k != 'safe_serialization'}
            model.save_pretrained(save_path, **fallback_kwargs)
            return True
        except Exception as e2:
            print(f"Error: Failed to save model completely: {e2}")
            return False

def _infer_hw(num_patches: int) -> tuple[int, int]:
    height = int(math.sqrt(num_patches))
    if height * height == num_patches:
        return height, height
    for height in range(height, 0, -1):
        if num_patches % height == 0:
            return height, num_patches // height
    raise ValueError(f"그리드 추정 실패: 패치 수={num_patches}")

# ---------------------------------------------------------------------------
# ‣ Simple MLP Blocks
# ---------------------------------------------------------------------------
class MLPResampler(nn.Module):
    """Flexible MLP Resampler with depth option: (입력: [B*V, S, Din]) → (출력: [B*V, S, Dout])"""
    def __init__(self, vision_dim: int, latent_dim: int, hidden_dim: Optional[int] = None, depth: int = 3, use_ln: bool = True):
        super().__init__()
        hd = hidden_dim or max(vision_dim, latent_dim)
        layers = []
        d_prev = vision_dim
        
        # Build layers based on depth
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_prev, hd))
            layers.append(nn.LayerNorm(hd, eps=1e-5) if use_ln else nn.Identity())
            layers.append(nn.GELU())
            d_prev = hd
        
        # Final output layer
        layers.append(nn.Linear(d_prev, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        orig_shape = vision_features.shape
        if vision_features.dim() == 3:
            _, _, D = vision_features.shape
            x = vision_features.reshape(-1, D)  # [B*V*S, D]
        else:
            x = vision_features
        
        x = self.mlp(x)
        
        if vision_features.dim() == 3:
            x = x.view(orig_shape[0], orig_shape[1], -1)  # [B*V, S, latent_dim]
        return x

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
# ‣ VICReg Loss
# ---------------------------------------------------------------------------
class VicRegLoss(nn.Module):
    """
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
        self.language_model = AutoModelForCausalLM.from_pretrained(lm_name)
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
        batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
        vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)

        if stage == "vision":
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

        # vision 외 단계: Resampler → LM
        resampled_features = self._extract_resampler_features(vision_hidden_states, batch_size, num_views)

        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            S, D = resampled_features.size(1), resampled_features.size(2)
            resampled_features = resampled_features.view(batch_size, num_views * S, D)
            return self._compute_autoregressive_loss(
                resampled_features, input_ids, attention_mask, labels
            )

        raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 32, temperature: float = 0.7, **kwargs):
        """간소화된 생성 함수"""
        self.eval()
        try:
            batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
            vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)
            vision_tokens = self._process_vision_features_for_generation(vision_hidden_states, batch_size, num_views)

            input_ids, attention_mask = self._prepare_generation_inputs(input_ids, attention_mask, batch_size, normalized_pixels.device)
            combined_inputs = self._create_combined_inputs_for_generation(vision_tokens, input_ids, attention_mask)
            generation_kwargs = self._build_generation_kwargs(combined_inputs, max_new_tokens, temperature, **kwargs)
            generated_ids = self.language_model.generate(**generation_kwargs)
            return self._postprocess_generated_text(generated_ids)
        except Exception as e:
            print(f"[Generate] Error in generation: {e}")
            import traceback; traceback.print_exc()
            return self._get_fallback_generation_result(pixel_values.size(0) if pixel_values.ndim in (4,5) else 1,
                                                        pixel_values.device if isinstance(pixel_values, torch.Tensor) else 'cpu')

    # ==================== 비전/텍스트 헬퍼 ====================
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

    def _process_vision_features_for_generation(self, vision_hidden_states, batch_size, num_views):
        resampled = self._extract_resampler_features(vision_hidden_states, batch_size, num_views)  # [B*V,S,D_latent]
        seq_len = resampled.size(1); feature_dim = resampled.size(2)
        resampled = resampled.view(batch_size, num_views * seq_len, feature_dim)
        vision_tokens = self._project_vision_tokens(resampled)
        return vision_tokens

    def _extract_resampler_features(self, vision_hidden_states, batch_size, num_views):
        """
        Resampler를 통한 vision feature 추출
        
        Args:
            vision_hidden_states: [B*V, S, D_vision] - Vision encoder의 출력
            batch_size: 배치 크기
            num_views: 뷰 개수
            
        Returns:
            resampled_features: [B*V, S, D_latent] - Resampler 출력 (projection 전)
        """
        # Resampler 통과
        resampled_features = self.resampler(vision_hidden_states)  # [B*V, S, D_latent]
        
        # Shape 검증
        expected_batch_size = batch_size * num_views
        if resampled_features.size(0) != expected_batch_size:
            raise ValueError(
                f"Resampler output batch dimension mismatch: "
                f"expected {expected_batch_size}, got {resampled_features.size(0)}"
            )
        
        return resampled_features

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
        DEFAULT_PROMPT = "Describe this panoramic image in detail."
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

    def _create_combined_inputs_for_generation(self, vision_tokens, input_ids, attention_mask):
        # 텍스트 임베딩
        if hasattr(self.language_model, 'get_input_embeddings'):
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        elif hasattr(self.language_model, 'base_model'):
            text_embeddings = self.language_model.base_model.get_input_embeddings()(input_ids)
        else:
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
        vision_attention = torch.ones(vision_tokens.size(0), vision_tokens.size(1),
                                      dtype=torch.long, device=vision_tokens.device)
        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)
        return {'inputs_embeds': combined_embeddings, 'attention_mask': combined_attention}

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

    def _postprocess_generated_text(self, generated_ids):
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned_texts = [t.strip() for t in generated_texts]
        return {"generated_ids": generated_ids, "text": cleaned_texts}

    def _get_fallback_generation_result(self, batch_size, device):
        fallback_text = ["a panoramic view"] * batch_size
        fallback_ids = torch.ones((batch_size, 5), dtype=torch.long, device=device)
        return {"generated_ids": fallback_ids, "text": fallback_text}

    # ==================== VICReg & AR 헬퍼 ====================
    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,   # [B*V, S, D_vicreg]
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,
    ):
        if num_views <= 1:
            return torch.zeros((), device=vision_output.device)

        has_cls_token = self._has_cls_token(vision_output)
        patch_features = vision_output[:, 1:] if has_cls_token else vision_output
        num_patches = patch_features.size(1)

        grid_height, grid_width = _infer_hw(num_patches)
        patch_features = patch_features.view(batch_size, num_views, grid_height, grid_width, -1)
        overlap_columns = max(1, int(grid_width * overlap_ratio))

        total_loss = 0.0
        num_pairs = 0

        for b in range(batch_size):
            batch_features = patch_features[b]  # [V, H, W, D]
            for v in range(num_views):
                nv = (v + 1) % num_views
                curr_right = batch_features[v, :, -overlap_columns:, :]
                next_left  = batch_features[nv, :, :overlap_columns, :]
                curr_flat = curr_right.reshape(-1, curr_right.shape[-1])
                next_flat = next_left.reshape(-1,  next_left.shape[-1])
                if curr_flat.shape[0] > 0:
                    pair_loss = self.vicreg_loss(curr_flat, next_flat)
                    if torch.isfinite(pair_loss):
                        total_loss += pair_loss
                        num_pairs += 1

        final_loss = total_loss / num_pairs if num_pairs > 0 else torch.zeros((), device=vision_output.device)
        return torch.clamp(final_loss, max=1e6)

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
        device = vision_tokens.device
        if text_inputs is not None:
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            labels = text_inputs.get('labels')
            batch_size = text_inputs['batch_size']
        else:
            batch_size = input_ids.size(0)

        if hasattr(self.language_model, 'get_input_embeddings'):
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        elif hasattr(self.language_model, 'base_model'):
            text_embeddings = self.language_model.base_model.get_input_embeddings()(input_ids)
        else:
            text_embeddings = self.language_model.model.embed_tokens(input_ids)

        if vision_tokens.size(0) != batch_size:
            min_batch = min(vision_tokens.size(0), batch_size)
            vision_tokens = vision_tokens[:min_batch]
            text_embeddings = text_embeddings[:min_batch]
            attention_mask = attention_mask[:min_batch]
            if labels is not None:
                labels = labels[:min_batch]
            batch_size = min_batch

        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
        vision_attention = torch.ones(batch_size, vision_tokens.size(1), dtype=torch.long, device=device)
        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)
        result = {'inputs_embeds': combined_embeddings, 'attention_mask': combined_attention}
        if labels is not None:
            vision_labels = torch.full((batch_size, vision_tokens.size(1)), self.ignore_index, dtype=labels.dtype, device=device)
            combined_labels = torch.cat([vision_labels, labels], dim=1)
            result['labels'] = combined_labels
        return result

    def _compute_autoregressive_loss(self, image_features, input_ids, attention_mask, labels):
        vision_tokens = self._project_vision_tokens(image_features)
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

    def _project_vision_tokens(self, image_features):
        return self.vision_to_language_projection(image_features)

    # ----- training helpers (text) -----
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
            if _safe_save_pretrained(self.language_model, str(save_dir), safe_serialization=True):
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

# -------------------- 편의: combined inputs 생성 (공용) --------------------
# def _stack_pad_mask(mask_list):  # 사용되지 않음
#     return torch.cat(mask_list, dim=1)

# 클래스 메서드에 두기 애매한 보조 함수들 보완
def _create_combined_inputs_for_training(self, vision_tokens, text_inputs):
    return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)

# 바인딩
PanoramaVLM._create_combined_inputs_for_training = _create_combined_inputs_for_training
