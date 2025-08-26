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
# ==================== Spherical Anchor Pooler ====================  ### NEW
class SphericalTokenPooler(nn.Module):
    """
    구면 앵커 기반 토큰 축소:
    - anchors: [K, 3] 단위벡터(구면 위 점)
    - forward 입력:
        patch_feats [Bv, S, D], patch_dirs [Bv, S, 3] (각 패치의 광선 방향, 정규화)
    - 출력:
        tokens [Bv, K, D], assign [Bv, S, K]
    """
    def __init__(self, K: int, learnable_anchors: bool = False, tau: float = 0.07, init_anchors=None):
        super().__init__()
        self.K = K
        self.tau = tau
        if init_anchors is None:
            anchors = torch.randn(K, 3)
            anchors = anchors / anchors.norm(dim=-1, keepdim=True)
        else:
            anchors = init_anchors / init_anchors.norm(dim=-1, keepdim=True)
        if learnable_anchors:
            self.anchors = nn.Parameter(anchors)
        else:
            self.register_buffer("anchors", anchors)

    def forward(self, patch_feats: torch.Tensor, patch_dirs: torch.Tensor):
        # patch_feats: [Bv, S, D], patch_dirs: [Bv, S, 3]
        # 코사인 유사도 기반 소프트 어사인
        sim = torch.einsum('bsd,kd->bsk', patch_dirs, self.anchors) / self.tau  # [Bv,S,K]
        assign = F.softmax(sim, dim=2)
        # 가중 평균 풀링: A^T * feats / sum A
        denom = assign.sum(dim=1, keepdim=True) + 1e-6  # [Bv,1,K]
        tokens = torch.einsum('bsk,bsd->bkd', assign, patch_feats) / denom.squeeze(1)  # [Bv,K,D]
        return tokens, assign

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
            _, _, dim = vision_features.shape
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
# ‣ VICReg Loss
# ---------------------------------------------------------------------------
class VicRegLoss(nn.Module):
    """
    VICReg (Bardes et al.)와 Meta VICRegL 코드의 스케일/정규화에 맞춘 구현
    - inv: MSE(x, y)
    - var: mean(ReLU(gamma - std)) for x,y 각각을 1/2 가중으로 합산
    - cov: off-diagonal squared 평균을 채널 수(D)로 나눠 정규화, x,y 1/2 평균
    - 선택: DDP 전역 통계(gather)로 var/cov 안정화
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
        # utils.gather_center(z)와 유사하게 모든 rank에서 모아 center할 수 있도록 확장
        # 환경에 맞춰 all_gather 구현을 넣으세요.
        return z  # placeholder: DDP 미사용이면 그대로

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        if N < 2:
            # 샘플이 너무 적으면 inv만
            return F.mse_loss(x, y) * self.similarity_weight

        # 1) Invariance (MSE)
        inv = F.mse_loss(x, y)

        # 2) Variance (per-dim std >= gamma)
        xg = self._gather_if_needed(x)
        yg = self._gather_if_needed(y)

        std_x = torch.sqrt(xg.var(dim=0, unbiased=False) + 1e-4)
        std_y = torch.sqrt(yg.var(dim=0, unbiased=False) + 1e-4)
        var = 0.5 * (F.relu(self.gamma - std_x).mean() + F.relu(self.gamma - std_y).mean())

        # 3) Covariance (off-diagonal suppression, normalized by D)
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
        # 하드 클리핑은 지양: 수렴을 왜곡합니다.
        if not torch.isfinite(total):
            total = torch.zeros((), device=x.device, dtype=x.dtype)
        return total


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
    
    def __init__(self, config=None, **kwargs):
        super().__init__()

        # 설정 시스템 통합 ------------------------------------------------
        if config is not None:
            self.config = config
            print(f"[Model] Using provided ModelConfig")
        else:
            # config가 없는 경우 kwargs나 기본값으로 config 생성
            from .config import ModelConfig
            self.config = ModelConfig(**kwargs)
            print(f"[Model] Created ModelConfig from kwargs")

        # 비전 인코더 초기화 ------------------------------------------------
        self.vision_encoder = AutoModel.from_pretrained(self.config.vision_name, trust_remote_code=True)
        if hasattr(self.vision_encoder, "vision_model"):
            self.vision_encoder = self.vision_encoder.vision_model
        vision_hidden_size = self._get_vision_hidden_size(self.vision_encoder)
        
        # VICReg 정규화 레이어 (원 VICReg 철학 준수를 위해 선택적 비활성화 가능)
        self.use_vicreg_norm = getattr(self.config, 'use_vicreg_norm', False)
        if self.use_vicreg_norm:
            self.vicreg_norm = nn.LayerNorm(vision_hidden_size)
        else:
            self.vicreg_norm = nn.Identity()
        
        # 리샘플러 초기화 ---------------------------------------------------
        if self.config.resampler_type == "mlp":
            self.resampler = MLPResampler(vision_hidden_size, self.config.latent_dimension)
        else:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {self.config.resampler_type}")

        # 언어 모델 초기화 ---------------------------------------------------
        self.language_model = AutoModelForCausalLM.from_pretrained(self.config.language_model_name)
        
        # 비전 특징을 언어 모델 임베딩 공간으로 투영하는 레이어
        self.vision_to_language_projection = nn.Linear(
            self.config.latent_dimension, 
            self.language_model.config.hidden_size
        )

        # 토크나이저 초기화 및 설정 개선 ------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_name)
        self._setup_tokenizer()
        
        # 특수 토큰 정의
        self.vision_token_id = self.tokenizer.convert_tokens_to_ids("<|vision|>")
        if self.vision_token_id == self.tokenizer.unk_token_id:
            self.vision_token_id = self.tokenizer.bos_token_id
        
        # UniversalTextFormatter 초기화 (생성 시 사용)
        try:
            from .processors.universal_text_formatter import UniversalTextFormatter
            self._text_formatter = UniversalTextFormatter(
                tokenizer_name_or_path=self.config.language_model_name,
                system_msg="You are an expert assistant specialized in analyzing panoramic images."
            )
            print(f"[Model] Initialized UniversalTextFormatter for {self._text_formatter.model_family}")
        except Exception as e:
            print(f"[Model] Failed to initialize UniversalTextFormatter: {e}")
            self._text_formatter = None

        # VICReg 손실 함수 및 가중치 / 중첩 비율 ---------------------------
        self.vicreg_loss = VicRegLoss(
            similarity_weight=self.config.vicreg_similarity_weight,
            variance_weight=self.config.vicreg_variance_weight,
            covariance_weight=self.config.vicreg_covariance_weight
        )
        
        
        # Config에서 설정값들 가져오기
        self.vicreg_loss_weight = self.config.vicreg_loss_weight
        self.vicreg_overlap_ratio = self.config.vicreg_overlap_ratio

        
        # 텍스트 처리 관련 설정
        self.max_text_length = self.config.max_text_length
        self.ignore_index = -100
        # num_views 경고 플래그
        self._warned_single_view = False
        # Loss 검증 디버깅 플래그 (개발/디버깅 시에만 활성화)
        self._debug_loss_verification = False
        

    def _setup_tokenizer(self):
        """토크나이저 설정 강화 - 모든 특수 토큰 안전 설정"""
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
    
    def _create_combined_inputs(self, vision_tokens, input_ids=None, attention_mask=None, labels=None, text_inputs=None):
        """통합된 비전-텍스트 입력 결합 (학습/생성 모두 지원)"""
        device = vision_tokens.device
        
        # text_inputs dict 또는 개별 인자들로 처리 가능
        if text_inputs is not None:
            # Training 모드: text_inputs dict 사용
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            labels = text_inputs.get('labels')
            batch_size = text_inputs['batch_size']
        else:
            # Generation 모드: 개별 인자 사용
            batch_size = input_ids.size(0)
        
        # 텍스트 임베딩 생성 (LoRA 호환)
        if hasattr(self.language_model, 'get_input_embeddings'):
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        elif hasattr(self.language_model, 'base_model'):
            # LoRA가 적용된 경우
            text_embeddings = self.language_model.base_model.get_input_embeddings()(input_ids)
        else:
            # 폴백
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        # 배치 크기 일치 확인
        if vision_tokens.size(0) != batch_size:
            min_batch = min(vision_tokens.size(0), batch_size)
            vision_tokens = vision_tokens[:min_batch]
            text_embeddings = text_embeddings[:min_batch]
            attention_mask = attention_mask[:min_batch]
            if labels is not None:
                labels = labels[:min_batch]
            batch_size = min_batch
        
        # 임베딩 결합
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
        
        # 어텐션 마스크 결합
        vision_attention = torch.ones(
            batch_size, vision_tokens.size(1), 
            dtype=torch.long, device=device
        )
        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)
        
        result = {
            'inputs_embeds': combined_embeddings,
            'attention_mask': combined_attention
        }
        
        # 학습용인 경우 labels 추가
        if labels is not None:
            vision_labels = torch.full(
                (batch_size, vision_tokens.size(1)), 
                self.ignore_index, 
                dtype=labels.dtype, 
                device=device
            )
            combined_labels = torch.cat([vision_labels, labels], dim=1)
            result['labels'] = combined_labels
        
        return result

    # ==================== 메인 순전파 및 생성 함수 ====================
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
        파노라마 VLM 순전파 - 3단계 학습 지원

        Args:
            pixel_values (torch.Tensor): 파노라마 이미지 픽셀 값.
                Shape: [B, V, C, H, W] 또는 [B, C, H, W].
                B: 배치 크기, V: 뷰 개수, C: 채널, H: 높이, W: 너비.
            input_ids (Optional[torch.Tensor]): 텍스트 토큰 ID. stage가 'resampler' 또는 'finetune'일 때 필수.
                Shape: [B, S_text]. S_text: 텍스트 시퀀스 길이.
            attention_mask (Optional[torch.Tensor]): 텍스트 어텐션 마스크.
                Shape: [B, S_text].
            labels (Optional[torch.Tensor]): 텍스트 레이블.
                Shape: [B, S_text].
            stage (str): 학습 단계. "vision" (VICReg), "resampler", "finetune" 중 하나.
            view_metadata (Optional[list]): VICReg-L을 위한 메타데이터 리스트.

        Returns:
            dict: 손실 및 관련 메트릭을 담은 딕셔너리.
                - stage="vision": {"loss": scalar, "vicreg_loss": scalar, ...}
                - stage="resampler" or "finetune": {"loss": scalar, "ar_loss": scalar, "logits": [B, S_combined, V_size]}
                  S_combined: 비전+텍스트 토큰 길이, V_size: 어휘 크기
        """
        # 통합된 전처리 사용
        batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
        vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)
        
        if stage == "vision":
            if num_views <= 1 and not self._warned_single_view:
                print("[VICReg] Warning: num_views <= 1, VICReg 손실이 0이 됩니다.")
                self._warned_single_view = True
            
            # VICReg 가중치 확인
            total_vicreg_weight = self.vicreg_loss_weight
            
            # 전체 가중치가 0.0이면 계산 건너뛰기
            if total_vicreg_weight == 0.0:
                if not hasattr(self, '_warned_zero_vicreg_weight'):
                    print("[VICReg] Warning: 모든 VICReg 가중치가 0.0입니다. VICReg 계산을 건너뜁니다.")
                    self._warned_zero_vicreg_weight = True
                
                return {
                    "loss": torch.zeros((), device=vision_hidden_states.device),
                    "vicreg_loss": torch.zeros((), device=vision_hidden_states.device),
                    "vicreg_raw": torch.zeros((), device=vision_hidden_states.device),
                    "vicreg_weight": self.vicreg_loss_weight
                }
            
            overlap_ratio = kwargs.get('overlap_ratio', self.vicreg_overlap_ratio)
            
            # VICReg 손실 계산
            vicreg_raw = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, overlap_ratio
            )
            vicreg_loss = vicreg_raw * self.vicreg_loss_weight
            
            total_loss = vicreg_loss
            
            loss_dict = {
                "loss": total_loss,
                "vicreg_loss": vicreg_loss,
                "vicreg_raw": vicreg_raw.detach(),
                "vicreg_weight": self.vicreg_loss_weight
            }
            return loss_dict
        
        to_resampler = vision_hidden_states

        resampled_features = self.resampler(to_resampler)  # [B*V, L, D]; L=K or S
        
        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            
            S, D = resampled_features.size(1), resampled_features.size(2)
            resampled_features = resampled_features.view(batch_size, num_views * S, D)
            
            return self._compute_autoregressive_loss(
                resampled_features, input_ids, attention_mask, labels
            )
        else:
            raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    # ==================== 비전 특징 처리 헬퍼 함수들 ====================
    def _convert_to_spatial_features(self, vision_hidden_states: torch.Tensor, 
                                   batch_size: int, num_views: int) -> torch.Tensor:
        del batch_size, num_views  # 현재는 사용하지 않음
        """
        패치 토큰을 공간 특징맵으로 변환
        
        Args:
            vision_hidden_states: [B*V, seq_len, hidden_dim] (seq_len includes CLS token)
            batch_size: 배치 크기
            num_views: 뷰 개수
            
        Returns:
            spatial_features: [B*V, C, H, W] 공간 특징맵
        """
        BV, _, C = vision_hidden_states.shape
        
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
        vision_output: torch.Tensor,
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,
    ):
        """
        VICReg Loss: 동일 배치 내 인접 뷰 간 겹치는 영역 정합
        
        겹치는 관계: view_0↔view_1, view_1↔view_2, ..., view_(V-1)↔view_0 (순환)
        각 배치에서 독립적으로 계산하여 올바른 파노라마 연속성 학습
        """
        if num_views <= 1:
            return torch.zeros((), device=vision_output.device)
        
        # 디버그 정보 (한 번만 출력)
        if not hasattr(self, '_debug_printed'):
            print(f"[VICReg Debug] vision_output shape: {vision_output.shape}")
            print(f"[VICReg Debug] batch_size: {batch_size}, num_views: {num_views}")
            self._debug_printed = True
        
        # CLS 토큰 제거 (있는 경우)
        has_cls_token = self._has_cls_token(vision_output)
        patch_features = vision_output[:, 1:] if has_cls_token else vision_output
        num_patches = patch_features.size(1)
        
        if not hasattr(self, '_debug_printed2'):
            print(f"[VICReg Debug] has_cls_token: {has_cls_token}, num_patches: {num_patches}")
        
        # [B*V, num_patches, hidden_dim] -> [B, V, H, W, hidden_dim]
        grid_height, grid_width = _infer_hw(num_patches)
        patch_features = patch_features.view(batch_size, num_views, grid_height, grid_width, -1)
        overlap_columns = max(1, int(grid_width * overlap_ratio))
        
        if not hasattr(self, '_debug_printed2'):
            print(f"[VICReg Debug] grid: {grid_height}x{grid_width}, overlap_columns: {overlap_columns}")
            self._debug_printed2 = True
        
        total_loss = 0.0
        num_pairs = 0
        
        # 각 배치에 대해 독립적으로 VICReg 계산
        for b in range(batch_size):
            batch_features = patch_features[b]  # [V, H, W, hidden_dim]
            
            # 인접 뷰 쌍들에 대해 겹치는 영역 추출
            for v in range(num_views):
                next_v = (v + 1) % num_views
                
                # 현재 뷰의 오른쪽 영역
                curr_right = batch_features[v, :, -overlap_columns:, :]  # [H, overlap_cols, D]
                
                # 다음 뷰의 왼쪽 영역  
                next_left = batch_features[next_v, :, :overlap_columns, :]  # [H, overlap_cols, D]
                
                # 플래튼화하여 VICReg 계산
                curr_flat = curr_right.reshape(-1, curr_right.shape[-1])  # [H*overlap_cols, D]
                next_flat = next_left.reshape(-1, next_left.shape[-1])   # [H*overlap_cols, D]
                
                if curr_flat.shape[0] > 0:  # 유효한 패치가 있을 때만
                    # VICReg Loss 계산
                    pair_loss = self.vicreg_loss(curr_flat, next_flat)
                    
                    # NaN/Inf 체크
                    if torch.isfinite(pair_loss):
                        total_loss += pair_loss
                        num_pairs += 1
        
        final_loss = total_loss / num_pairs if num_pairs > 0 else torch.zeros((), device=vision_output.device)
        final_loss = torch.clamp(final_loss, max=1e6)  # 최종 클리핑
        
        if not hasattr(self, '_debug_printed4'):
            print(f"[VICReg Debug] Final VICReg loss: {final_loss.item():.6f} (pairs: {num_pairs})")
            self._debug_printed4 = True
            
        return final_loss





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
                 max_new_tokens: int = 32, temperature: float = 0.7,
                 view_metadata: Optional[list] = None,  # <- 추가  ### NEW
                 **kwargs):
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
            vision_tokens = self._process_vision_features_for_generation(
                vision_hidden_states, batch_size, num_views, view_metadata=view_metadata  # ### NEW
            )
            
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
        to_resampler = vision_hidden_states

        resampled = self.resampler(to_resampler)                  # [B*V,L,D]
        seq_len = resampled.size(1); feature_dim = resampled.size(2)
        resampled = resampled.view(batch_size, num_views * seq_len, feature_dim)

        # 3. 언어 모델 임베딩 공간으로 투영
        vision_tokens = self._project_vision_tokens(resampled)
        return vision_tokens
    
    # ==================== 생성용 입력 준비 헬퍼 함수들 ====================
    def _prepare_generation_inputs(self, input_ids, attention_mask, batch_size, device):
        """생성을 위한 텍스트 입력 준비"""
        if input_ids is None:
            print("[Generate] Warning: input_ids not provided, creating default prompt for captioning")
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