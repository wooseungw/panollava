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
    """전역 재현성(가능한 한)을 위한 시드 고정 유틸리티.
    Args:
        seed: 시드 값
        deterministic: True이면 cudnn deterministic 모드 활성 (성능 ↓ 가능)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: F401
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
    """
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
    """
    주어진 패치 토큰 개수(`num_patches`)로부터 (H, W) 그리드 크기를 추정합니다.
    1) 완전제곱 → 정사각형  
    2) 아니면 √N 이하에서 가장 큰 약수를 찾아 (height, width = N//height) 반환
    """
    height = int(math.sqrt(num_patches))
    if height * height == num_patches:
        return height, height
    for h in range(height, 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    raise ValueError(f"그리드 추정 실패: 패치 수={num_patches}")


# ==================== Resampler ====================
class MLPResampler(nn.Module):
    """다층 퍼셉트론을 사용한 리샘플러
    입력: (B*V, L, Dv) 또는 (B, T, Dv) 또는 (N, Dv)
    출력: 동일한 첫 두 차원 유지, 마지막 차원만 latent_dim으로 변환
    """

    def __init__(self, vision_dim: int, latent_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_dim, latent_dim)
        self.bn1 = nn.LayerNorm(latent_dim, eps=1e-5)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        orig_shape = vision_features.shape
        if vision_features.dim() == 3:
            n, l, d = orig_shape
            x = vision_features.reshape(-1, d)  # (n*l, d)
        else:
            x = vision_features  # (n, d)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.linear2(x)
        if vision_features.dim() == 3:
            x = x.view(n, l, -1)
        return x


# ==================== VICReg Loss ====================
class VicRegLoss(nn.Module):
    """
    VICReg (Bardes et al.) + Meta VICRegL 코드 스타일의 안정적 구현
    - inv: MSE(x, y)
    - var: mean(ReLU(gamma - std)) for x,y 각각 1/2 가중
    - cov: off-diagonal 제곱합 / D, x,y 평균
    """

    def __init__(
        self,
        similarity_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        gamma: float = 1.0,
        use_ddp_gather: bool = False,
    ):
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
        # DDP 환경이면 all_gather/concat 후 center를 적용하도록 확장 가능
        return z

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (N, D)
        # y: (N, D)
        # returns: (1,)
        N, D = x.shape
        if N < 2:
            return self.similarity_weight * F.mse_loss(x, y)

        # 1) invariance
        inv = F.mse_loss(x, y,reduction='mean')

        # 2) variance
        xg = self._gather_if_needed(x)
        yg = self._gather_if_needed(y)
        std_x = torch.sqrt(xg.var(dim=0, unbiased=False) + 1e-4)
        std_y = torch.sqrt(yg.var(dim=0, unbiased=False) + 1e-4)
        var = 0.5 * (F.relu(self.gamma - std_x).mean() + F.relu(self.gamma - std_y).mean())

        # 3) covariance
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


# ==================== PanoramaVLM (VICReg + AR) ====================
class PanoramaVLM(nn.Module):
    """파노라마 비전-언어 모델

    3단계 학습 구조:
    1) Vision: VICReg을 통한 파노라마 뷰 간 중첩 영역 일관성 학습
    2) Resampler: 리샘플러 + 투영 레이어 학습 (AR loss)
    3) Finetune: 전체 모델 파인튜닝 (AR loss)
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()

        # ---- 설정 처리
        if config is not None:
            self.config = config
            print(f"[Model] Using provided ModelConfig")
        else:
            try:
                from .config import ModelConfig  # type: ignore
                self.config = ModelConfig(**kwargs)
                print(f"[Model] Created ModelConfig from kwargs")
            except Exception:
                # 간단 폴백
                class _Cfg:
                    def __init__(self, **kw):
                        self.vision_name = kw.get("vision_name", "google/siglip-base-patch16-224")
                        self.language_model_name = kw.get("language_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
                        self.resampler_type = kw.get("resampler_type", "mlp")
                        self.latent_dimension = kw.get("latent_dimension", 768)
                        self.vicreg_similarity_weight = kw.get("vicreg_similarity_weight", 25.0)
                        self.vicreg_variance_weight = kw.get("vicreg_variance_weight", 25.0)
                        self.vicreg_covariance_weight = kw.get("vicreg_covariance_weight", 1.0)
                        self.vicreg_loss_weight = kw.get("vicreg_loss_weight", 1.0)
                        self.vicreg_overlap_ratio = kw.get("vicreg_overlap_ratio", 0.5)
                        self.max_text_length = kw.get("max_text_length", 512)
                        self.use_vicreg_norm = kw.get("use_vicreg_norm", False)
                self.config = _Cfg(**kwargs)
                print(f"[Model] Using fallback inline config")

        # ---- 비전 인코더
        self.vision_encoder = AutoModel.from_pretrained(self.config.vision_name, trust_remote_code=True)
        if hasattr(self.vision_encoder, "vision_model"):
            self.vision_encoder = self.vision_encoder.vision_model
        vision_hidden_size = self._get_vision_hidden_size(self.vision_encoder)

        # VICReg 정규화 (선택)
        self.use_vicreg_norm = getattr(self.config, 'use_vicreg_norm', False)
        self.vicreg_norm = nn.LayerNorm(vision_hidden_size) if self.use_vicreg_norm else nn.Identity()

        # ---- 리샘플러
        if self.config.resampler_type == "mlp":
            self.resampler = MLPResampler(vision_hidden_size, self.config.latent_dimension)
        else:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {self.config.resampler_type}")

        # ---- 언어 모델 & 토크나이저
        self.language_model = AutoModelForCausalLM.from_pretrained(self.config.language_model_name)
        self.vision_to_language_projection = nn.Linear(
            self.config.latent_dimension,
            self.language_model.config.hidden_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_name)
        self._setup_tokenizer()
        self.vision_token_id = self.tokenizer.convert_tokens_to_ids("<|vision|>")
        if self.vision_token_id == self.tokenizer.unk_token_id:
            self.vision_token_id = self.tokenizer.bos_token_id

        # ---- UniversalTextFormatter (옵션)
        try:
            from .processors.universal_text_formatter import UniversalTextFormatter  # type: ignore
            self._text_formatter = UniversalTextFormatter(
                tokenizer_name_or_path=self.config.language_model_name,
                system_msg="You are an expert assistant specialized in analyzing panoramic images."
            )
            print(f"[Model] Initialized UniversalTextFormatter for {self._text_formatter.model_family}")
        except Exception as e:
            print(f"[Model] UniversalTextFormatter unavailable: {e}")
            self._text_formatter = None

        # ---- VICReg 손실 및 하이퍼파라미터
        self.vicreg_loss = VicRegLoss(
            similarity_weight=self.config.vicreg_similarity_weight,
            variance_weight=self.config.vicreg_variance_weight,
            covariance_weight=self.config.vicreg_covariance_weight
        )
        self.vicreg_loss_weight = self.config.vicreg_loss_weight
        self.vicreg_overlap_ratio = self.config.vicreg_overlap_ratio

        # ---- 기타 설정
        self.max_text_length = self.config.max_text_length
        self.ignore_index = -100
        self._warned_single_view = False
        self._debug_loss_verification = False

    # ---------------- 토크나이저 설정 ----------------
    def _setup_tokenizer(self):
        tokens_added = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[Tokenizer Setup] pad_token = eos_token: '{self.tokenizer.eos_token}'")
            else:
                self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'})
                tokens_added = True
                print(f"[Tokenizer Setup] Added eos/pad '<|endoftext|>'")

        # 필수 추가 토큰
        special_tokens_to_add = []
        vision_token = '<|vision|>'
        if not any(vision_token in str(t) for t in self.tokenizer.additional_special_tokens):
            special_tokens_to_add.append(vision_token)
        if self.tokenizer.eos_token != '<|endoftext|>':
            if not any('<|endoftext|>' in str(t) for t in self.tokenizer.additional_special_tokens):
                special_tokens_to_add.append('<|endoftext|>')

        if special_tokens_to_add:
            added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
            tokens_added = tokens_added or (added > 0)
            print(f"[Tokenizer Setup] Added special tokens: {special_tokens_to_add}")

        if tokens_added:
            old_n = self.language_model.get_input_embeddings().weight.size(0)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            new_n = self.language_model.get_input_embeddings().weight.size(0)
            print(f"[Tokenizer Setup] Resize embeddings: {old_n} -> {new_n}")

        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            print(f"[Tokenizer Setup] ✓ pad_token_id == eos_token_id")

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    # ---------------- 유틸 ----------------
    @staticmethod
    def _get_vision_hidden_size(vision_model: nn.Module) -> int:
        possible = ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]
        for key in possible:
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
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'batch_size': batch_size
        }

    def _build_combined_inputs(
        self,
        vision_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """비전 임베딩과 텍스트 임베딩을 결합해 CausalLM 입력 형식으로 변환"""
        device = vision_tokens.device
        batch_size = input_ids.size(0)

        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 텍스트 임베딩
        if hasattr(self.language_model, 'get_input_embeddings'):
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        elif hasattr(self.language_model, 'base_model'):
            text_embeddings = self.language_model.base_model.get_input_embeddings()(input_ids)  # type: ignore[attr-defined]
        else:
            text_embeddings = self.language_model.model.embed_tokens(input_ids)  # type: ignore[attr-defined]

        if vision_tokens.size(0) != batch_size:
            min_b = min(vision_tokens.size(0), batch_size)
            vision_tokens = vision_tokens[:min_b]
            text_embeddings = text_embeddings[:min_b]
            attention_mask = attention_mask[:min_b]
            if labels is not None:
                labels = labels[:min_b]
            batch_size = min_b

        inputs_embeds = torch.cat([vision_tokens, text_embeddings], dim=1)
        vision_attention = torch.ones(batch_size, vision_tokens.size(1), dtype=torch.long, device=device)
        attn_mask = torch.cat([vision_attention, attention_mask], dim=1)

        result = {'inputs_embeds': inputs_embeds, 'attention_mask': attn_mask}
        if labels is not None:
            ignore = torch.full((batch_size, vision_tokens.size(1)), self.ignore_index, dtype=labels.dtype, device=device)
            result['labels'] = torch.cat([ignore, labels], dim=1)
        return result

    # ==================== 순전파 ====================
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
            pixel_values: [B, V, C, H, W] 또는 [B, C, H, W]
            stage: "vision" | "resampler" | "finetune"
        """
        batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
        vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)

        if stage == "vision":
            if num_views <= 1 and not self._warned_single_view:
                print("[VICReg] Warning: num_views <= 1, VICReg 손실이 0이 됩니다.")
                self._warned_single_view = True

            if self.vicreg_loss_weight == 0.0:
                if not hasattr(self, '_warned_zero_vicreg_weight'):
                    print("[VICReg] Warning: VICReg 가중치가 0.0입니다. 계산을 건너뜁니다.")
                    self._warned_zero_vicreg_weight = True
                zero = torch.zeros((), device=vision_hidden_states.device)
                return {"loss": zero, "vicreg_loss": zero, "vicreg_raw": zero, "vicreg_weight": 0.0}

            vicreg_raw = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, self.vicreg_overlap_ratio
            )
            vicreg_loss = vicreg_raw * self.vicreg_loss_weight
            return {
                "loss": vicreg_loss,
                "vicreg_loss": vicreg_loss,
                "vicreg_raw": vicreg_raw.detach(),
                "vicreg_weight": self.vicreg_loss_weight
            }

        # "resampler" | "finetune"
        to_resampler = vision_hidden_states  # [B*V, L, Dv]
        resampled = self.resampler(to_resampler)                 # [B*V, L, Dl]
        L = resampled.size(1); Dl = resampled.size(2)
        resampled = resampled.view(batch_size, num_views * L, Dl)  # [B, V*L, Dl]

        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            return self._compute_autoregressive_loss(
                resampled, input_ids, attention_mask, labels
            )

        raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    # -------- 비전 특징 헬퍼 --------
    def _normalize_pixel_values(self, pixel_values):
        if pixel_values.ndim == 5:
            B, V, _, _, _ = pixel_values.shape
        elif pixel_values.ndim == 4:
            B, _, _, _ = pixel_values.shape
            V = 1
            pixel_values = pixel_values.unsqueeze(1)
        else:
            raise ValueError(f"pixel_values shape invalid: {pixel_values.shape}")
        return B, V, pixel_values

    def _extract_vision_features(self, pixel_values, batch_size, num_views):
        flat = pixel_values.view(batch_size * num_views, *pixel_values.shape[2:])
        vision_out = self.vision_encoder(pixel_values=flat, return_dict=True)
        hidden = vision_out.last_hidden_state  # (B*V, P, D)
        hidden = self.vicreg_norm(hidden)
        return hidden

    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,
    ):
        """인접 뷰의 겹치는 패치 열에 대해 VICReg 계산"""
        # B: 배치 크기, V: 파노라마 뷰 개수, P: 비전 토큰(패치) 개수, D: 비전 특징 차원

        if num_views <= 1:
            return torch.zeros((), device=vision_output.device) # shape: () (스칼라)

        # 1. CLS 토큰 제거
        has_cls = self._has_cls_token(vision_output)
        # CLS 토큰이 있으면 첫번째 토큰을 제거합니다.
        # patch_feats shape: (B*V, P-1, D) 또는 (B*V, P, D)
        patch_feats = vision_output[:, 1:] if has_cls else vision_output
        
        num_patches = patch_feats.size(1) # P' = P-1 또는 P
        H, W = _infer_hw(num_patches)     # H * W == P'가 되는 H, W 계산

        # 2. 텐서 Reshape
        # 뷰와 패치 그리드(H, W)를 별도 차원으로 분리합니다.
        # patch_feats shape: (B, V, H, W, D)
        patch_feats = patch_feats.view(batch_size, num_views, H, W, -1)
        
        # 겹치는 영역의 너비(컬럼 수)를 계산합니다.
        overlap_cols = max(1, int(W * overlap_ratio)) # oc (스칼라)

        total = 0.0
        pairs = 0
        # 배치(B)와 뷰(V)를 순회하며 loss를 계산합니다.
        for b in range(batch_size):
            # feats shape: (V, H, W, D)
            feats = patch_feats[b]
            for v in range(num_views):
                nv = (v + 1) % num_views # 다음 뷰 인덱스
                
                # 3. 인접 뷰에서 겹치는 영역(패치) 추출
                # 현재 뷰의 오른쪽 영역
                # right shape: (H, oc, D)
                right = feats[v, :, -overlap_cols:, :]
                # 다음 뷰의 왼쪽 영역
                # left shape: (H, oc, D)
                left  = feats[nv, :, :overlap_cols, :]
                
                # 4. VICReg Loss 계산을 위해 2D로 Reshape
                # x, y shape: (H * oc, D)
                x = right.reshape(-1, right.shape[-1])
                y = left.reshape(-1, left.shape[-1])
                
                if x.numel() == 0 or y.numel() == 0:
                    continue
                
                # vicreg_loss는 두 (N, D) 텐서를 받아 스칼라 loss를 반환합니다.
                # loss shape: () (스칼라)
                loss = self.vicreg_loss(x, y)
                
                if torch.isfinite(loss):
                    total += loss
                    pairs += 1
                    
        # 5. 최종 Loss 계산
        # final shape: () (스칼라)
        final = total / pairs if pairs > 0 else torch.zeros((), device=vision_output.device)
        
        # 최종 반환 shape: () (스칼라)
        return final

    # -------- AR Loss --------
    def _project_vision_tokens(self, image_features: torch.Tensor) -> torch.Tensor:
        """비전 특징을 언어모델 임베딩 공간으로 투영"""
        return self.vision_to_language_projection(image_features)

    def _compute_autoregressive_loss(self, image_features, input_ids, attention_mask, labels):
        # 1) 비전 → LM 공간
        vision_tokens = self._project_vision_tokens(image_features)

        # 2) 텍스트 전처리
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels)

        # 3) 결합
        combined = self._build_combined_inputs(
            vision_tokens=vision_tokens,
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            labels=text_inputs['labels'],
        )

        # 4) LM forward
        try:
            outputs = self.language_model(
                inputs_embeds=combined['inputs_embeds'],
                attention_mask=combined['attention_mask'],
                labels=combined['labels'],
                return_dict=True
            )
            if not torch.isfinite(outputs.loss):
                manual = self._compute_manual_cross_entropy_loss(outputs.logits, combined['labels'])
                manual.setdefault('ar_loss', manual['loss'])
                return manual

            if self._debug_loss_verification:
                manual = self._compute_manual_cross_entropy_loss(outputs.logits, combined['labels'])
                diff = abs(outputs.loss.item() - manual['loss'].item())
                if diff > 0.01:
                    print(f"[Loss Debug] HF:{outputs.loss.item():.6f}, Manual:{manual['loss'].item():.6f}, Diff:{diff:.6f}")

            return {"loss": outputs.loss, "ar_loss": outputs.loss, "logits": outputs.logits}
        except Exception:
            fallback = torch.tensor(0.0, device=vision_tokens.device, requires_grad=True)
            return {
                "loss": fallback,
                "ar_loss": fallback,
                "logits": torch.zeros(
                    (text_inputs['batch_size'],
                     combined['inputs_embeds'].size(1),
                     self.language_model.config.vocab_size),
                    device=vision_tokens.device
                )
            }

    # ==================== 생성 (Generate) ====================
    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        간소화된 파노라마 이미지 텍스트 생성 함수
        """
        self.eval()
        try:
            # 1) 비전 특징 추출
            B, V, norm_pixels = self._normalize_pixel_values(pixel_values)
            hidden = self._extract_vision_features(norm_pixels, B, V)
            resampled = self.resampler(hidden)                      # [B*V, L, Dl]
            L = resampled.size(1); Dl = resampled.size(2)
            resampled = resampled.view(B, V * L, Dl)                # [B, V*L, Dl]
            vision_tokens = self._project_vision_tokens(resampled)  # [B, Tv, H_lm]

            # 2) 텍스트 입력 준비
            input_ids, attention_mask = self._prepare_generation_inputs(input_ids, attention_mask, B, norm_pixels.device)

            # 3) 결합 & 생성
            combined = self._build_combined_inputs(vision_tokens, input_ids, attention_mask, labels=None)
            gen_kwargs = self._build_generation_kwargs(combined, max_new_tokens, temperature, **kwargs)
            generated_ids = self.language_model.generate(**gen_kwargs)

            # 4) 디코드
            return self._postprocess_generated_text(generated_ids)

        except Exception as e:
            print(f"[Generate] Error: {e}")
            import traceback; traceback.print_exc()
            return self._get_fallback_generation_result(pixel_values.size(0) if pixel_values.ndim in (4,5) else 1,
                                                        pixel_values.device)

    # -------- 생성 보조 --------
    def _prepare_generation_inputs(self, input_ids, attention_mask, batch_size, device):
        if input_ids is None:
            enc = self.tokenizer(
                "Describe this panoramic image in detail.",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else (input_ids != self.tokenizer.pad_token_id).long()
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
                cfg = self.language_model.config
            elif hasattr(self.language_model, 'base_model') and hasattr(self.language_model.base_model, 'config'):
                cfg = self.language_model.base_model.config  # type: ignore[attr-defined]
            else:
                cfg = None
            if cfg:
                return {
                    'pad_token_id': getattr(cfg, 'pad_token_id', None),
                    'eos_token_id': getattr(cfg, 'eos_token_id', None),
                    'bos_token_id': getattr(cfg, 'bos_token_id', None)
                }
            return {}
        except Exception:
            return {}

    def _build_generation_kwargs(self, combined_inputs, max_new_tokens, temperature, **kwargs):
        temperature = max(0.1, min(temperature, 1.0))
        tokenizer_info = self._get_tokenizer_info()
        gen_kwargs = {
            **combined_inputs,  # inputs_embeds & attention_mask
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': kwargs.get('min_new_tokens', 5),
            'temperature': temperature,
            'do_sample': temperature > 0.1,
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 50),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.1),
            'length_penalty': kwargs.get('length_penalty', 1.0),
        }
        if tokenizer_info.get('pad_token_id') is not None:
            gen_kwargs['pad_token_id'] = tokenizer_info['pad_token_id']
        if tokenizer_info.get('eos_token_id') is not None:
            gen_kwargs['eos_token_id'] = tokenizer_info['eos_token_id']
        for key in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]
        return gen_kwargs

    def _postprocess_generated_text(self, generated_ids):
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if self._text_formatter is not None:
            cleaned = [self._text_formatter.extract_assistant_response(t) for t in texts]
        else:
            cleaned = [t.strip() for t in texts]
        return {"generated_ids": generated_ids, "text": cleaned}

    def _get_fallback_generation_result(self, batch_size, device):
        fallback_text = ["a panoramic view"] * batch_size
        fallback_ids = torch.ones((batch_size, 5), dtype=torch.long, device=device)
        return {"generated_ids": fallback_ids, "text": fallback_text}

    # -------- 수동 CE (디버그) --------
    def _compute_manual_cross_entropy_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not hasattr(self, '_debug_shift_logged'):
            valid = (shift_labels != self.ignore_index).sum()
            total = shift_labels.numel()
            print(f"[Loss Debug] logits:{logits.shape} labels:{labels.shape} valid:{valid.item()}/{total}")
            self._debug_shift_logged = True
        return {
            "loss": loss,
            "ar_loss": loss,
            "perplexity": torch.exp(loss) if torch.isfinite(loss) else torch.tensor(float('inf'))
        }

    # ==================== LoRA 지원 ====================
    def setup_lora_for_finetune(self, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
                                target_modules: Optional[list] = None) -> bool:
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. LoRA setup skipped.")
            return False
        try:
            if target_modules is None:
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            if hasattr(self.language_model, 'enable_input_require_grads'):
                self.language_model.enable_input_require_grads()
            trainable = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.language_model.parameters())
            print(f"✓ LoRA setup: trainable {trainable:,}/{total:,} ({trainable/total*100:.2f}%)")
            return True
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            return False

    def load_lora_weights(self, load_path: str):
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot load LoRA weights.")
            return False
        try:
            from peft import PeftModel  # type: ignore
            is_peft = hasattr(self.language_model, 'peft_config')
            if is_peft:
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
                from safetensors.torch import load_file  # type: ignore
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

    # ==================== 체크포인트/저장 ====================
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, lora_weights_path: Optional[str] = None,
                        device: str = "auto", auto_detect_lora: bool = True,
                        strict_loading: bool = False, **model_kwargs) -> 'PanoramaVLM':
        import torch
        from pathlib import Path

        print(f"🚀 PanoramaVLM 체크포인트 로딩: {checkpoint_path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

        print("📂 체크포인트 로딩 중...")
        ckpt = torch.load(ckpt_path, map_location=dev)
        hparams = ckpt.get('hyper_parameters', {})
        state_dict = ckpt.get('state_dict', {})

        # 설정 복원 (가능하면 사용자 config 시스템 사용)
        try:
            from .config import ConfigManager, ModelConfig  # type: ignore
            detected_config = ConfigManager.auto_detect_config(ckpt_path)
            if detected_config:
                print("🔍 설정 파일 자동 감지 성공")
                model_config = detected_config
            else:
                print("🔍 설정 파일 감지 실패 - 하이퍼파라미터에서 생성")
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
            if model_kwargs:
                print(f"🛠️ 사용자 파라미터 오버라이드: {list(model_kwargs.keys())}")
                model_config = model_config.update(**model_kwargs)
            model_params = model_config.get_model_kwargs()
            model_params['config'] = model_config
        except Exception as e:
            print(f"⚠️ 설정 시스템 사용 실패 ({e}) - 폴백 파라미터 사용")
            model_params = {
                'vision_name': hparams.get('vision_name', 'google/siglip-base-patch16-224'),
                'language_model_name': hparams.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'),
                'resampler_type': hparams.get('resampler_type', 'mlp'),
                'latent_dimension': hparams.get('latent_dimension', 768),
                'vicreg_loss_weight': hparams.get('vicreg_loss_weight', 1.0),
                'vicreg_overlap_ratio': hparams.get('vicreg_overlap_ratio', 0.5),
                'max_text_length': hparams.get('max_text_length', 512),
            }
            model_params.update(model_kwargs)

        print("🏗️  모델 인스턴스 생성 중...")
        model = cls(**model_params)

        print("⚙️  가중치 로딩 중...")
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                cleaned[k[6:]] = v  # 'model.' 제거
        if cleaned:
            missing, unexpected = model.load_state_dict(cleaned, strict=strict_loading)
            print(f"   - 로드된 키: {len(cleaned) - len(missing)}")
            if missing:
                print(f"   - 누락된 키: {len(missing)} (예: {missing[:3]}{'...' if len(missing) > 3 else ''})")
            if unexpected:
                print(f"   - 예상치 못한 키: {len(unexpected)} (예: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''})")
        else:
            print("   ⚠️  모델 가중치를 찾을 수 없음. 기본 초기화 사용.")

        # LoRA 자동 감지
        if auto_detect_lora and lora_weights_path is None:
            pot = ckpt_path.parent / "lora_weights"
            if pot.exists() and pot.is_dir():
                lora_weights_path = str(pot)
                print(f"🔍 LoRA 가중치 자동 감지: {lora_weights_path}")

        if lora_weights_path:
            from pathlib import Path as _P
            if _P(lora_weights_path).exists():
                print(f"🔧 LoRA 가중치 로딩: {lora_weights_path}")
                ok = model.load_lora_weights(str(lora_weights_path))
                print("   ✅ LoRA 로딩 성공" if ok else "   ❌ LoRA 로딩 실패")
            else:
                print(f"   ⚠️  LoRA 경로가 존재하지 않습니다: {lora_weights_path}")

        model = model.to(dev)
        model.eval()

        if not hasattr(model, 'tokenizer'):
            try:
                model.tokenizer = AutoTokenizer.from_pretrained(model_params.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'))
                print(f"   ✅ 토크나이저 로드 완료")
            except Exception as e:
                print(f"   ⚠️ 토크나이저 로드 실패: {e}")
        print(f"✅ 모델 로딩 완료 - Device: {device}")
        return model

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "auto", **kwargs) -> 'PanoramaVLM':
        from pathlib import Path
        path = Path(model_path)
        if path.is_dir():
            for cand in ["best.ckpt", "last.ckpt", "model.safetensors", "model_final.safetensors", "pytorch_model.bin"]:
                p = path / cand
                if p.exists():
                    return cls.from_checkpoint(str(p), device=device, **kwargs)
            raise FileNotFoundError(f"지원되는 모델 파일을 찾을 수 없습니다: {model_path}")
        else:
            return cls.from_checkpoint(str(path), device=device, **kwargs)

    def save_pretrained(self, save_directory: str, save_lora_separately: bool = True):
        from pathlib import Path
        import json
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 모델 저장 중: {save_directory}")

        # 가중치 저장
        try:
            from safetensors.torch import save_file  # type: ignore
            model_path = save_dir / "model.safetensors"
            save_file(self.state_dict(), model_path)
            print(f"   ✅ 모델 가중치 저장 (SafeTensors): {model_path}")
        except Exception:
            model_path = save_dir / "pytorch_model.bin"
            torch.save(self.state_dict(), model_path)
            print(f"   ✅ 모델 가중치 저장 (PyTorch): {model_path}")

        # 설정 저장
        try:
            if hasattr(self, 'config') and self.config:
                try:
                    updated = self.config.update(  # type: ignore[attr-defined]
                        vision_name=getattr(self.vision_encoder.config, 'name_or_path', getattr(self.config, 'vision_name', 'unknown')),
                        language_model_name=getattr(self.language_model.config, 'name_or_path', getattr(self.config, 'language_model_name', 'unknown')),
                        latent_dimension=self.vision_to_language_projection.in_features,
                        max_text_length=self.max_text_length,
                        vicreg_loss_weight=self.vicreg_loss_weight,
                        vicreg_overlap_ratio=self.vicreg_overlap_ratio,
                        description=f"Saved at {save_directory}"
                    )
                    config_path = save_dir / "model_config.json"
                    updated.save(config_path)  # type: ignore[attr-defined]
                    print(f"   ✅ ModelConfig 저장: {config_path}")
                except Exception:
                    # 폴백 JSON
                    cfg = {
                        "model_type": "PanoramaVLM",
                        "vision_name": getattr(self.vision_encoder.config, 'name_or_path', 'unknown'),
                        "language_model_name": getattr(self.language_model.config, 'name_or_path', 'unknown'),
                        "latent_dimension": self.vision_to_language_projection.in_features,
                        "max_text_length": self.max_text_length,
                        "vicreg_loss_weight": self.vicreg_loss_weight,
                        "vicreg_overlap_ratio": self.vicreg_overlap_ratio,
                    }
                    with open(save_dir / "config.json", 'w', encoding='utf-8') as f:
                        json.dump(cfg, f, indent=2, ensure_ascii=False)
                    print(f"   ✅ 설정 저장 (fallback): {save_dir/'config.json'}")
            else:
                cfg = {
                    "model_type": "PanoramaVLM",
                    "vision_name": getattr(self.vision_encoder.config, 'name_or_path', 'unknown'),
                    "language_model_name": getattr(self.language_model.config, 'name_or_path', 'unknown'),
                    "latent_dimension": self.vision_to_language_projection.in_features,
                    "max_text_length": self.max_text_length,
                    "vicreg_loss_weight": self.vicreg_loss_weight,
                    "vicreg_overlap_ratio": self.vicreg_overlap_ratio,
                }
                with open(save_dir / "config.json", 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(f"   ✅ 설정 저장: {save_dir/'config.json'}")
        except Exception as e:
            print(f"   ⚠️ 설정 저장 실패: {e}")

        if save_lora_separately:
            info = self.get_lora_info()
            if info.get("is_lora_enabled", False):
                lora_dir = save_dir / "lora_weights"
                ok = self.save_lora_weights(str(lora_dir))
                print(f"   {'✅' if ok else '⚠️'} LoRA 가중치 저장: {lora_dir}")

        print("🎉 모델 저장 완료")

    @staticmethod
    def create_model_factory(checkpoint_path: str, **default_kwargs):
        def factory(**kwargs):
            merged = {**default_kwargs, **kwargs}
            return PanoramaVLM.from_checkpoint(checkpoint_path, **merged)
        return factory
