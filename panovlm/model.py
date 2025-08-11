# coding: utf-8

import math
from typing import Optional, Dict, Any

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
        self.bn1 = nn.BatchNorm1d(latent_dim, eps=1e-5)
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
# ‣ VICReg Loss
# ---------------------------------------------------------------------------
class VicRegLoss(nn.Module):
    """VICReg 손실 함수: 분산-불변-공분산 정규화
    
    Args:
        similarity_weight: 유사성 손실 가중치 (기본값: 25.0)
        variance_weight: 분산 손실 가중치 (기본값: 25.0)  
        covariance_weight: 공분산 손실 가중치 (기본값: 1.0)
    """
    def __init__(self, similarity_weight: float = 25.0, variance_weight: float = 25.0, covariance_weight: float = 1.0):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight


    @staticmethod
    def _off_diagonal(x):
        # 공식 구현: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _variance_loss(self, x, eps=1e-4):
        # 공식 구현: 각 차원별 표준편차가 1 이상이 되도록
        std = torch.sqrt(x.var(dim=0) + eps)
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
        """VICReg 공식 구현 스타일 손실 계산
        Args:
            x: 첫 번째 임베딩 (N, D...)
            y: 두 번째 임베딩 (N, D...)
        Returns:
            total_loss: 전체 VICReg 손실
        """
        # 배치 차원으로 평면화 (B, ...) -> (B, D)
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))
        
        # VICReg 공식 구현: 표준화 (평균=0, 분산=1)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-6)
        
        # 1) Invariance(유사성) 손실: MSE
        sim_loss = F.mse_loss(x, y)
        # 2) Variance(분산) 손실: 각 차원별 std가 1 이상이 되도록
        var_loss = self._variance_loss(x) + self._variance_loss(y)
        # 3) Covariance(공분산) 손실: 오프다이애고널 제곱 평균
        cov_loss = self._covariance_loss(x) + self._covariance_loss(y)
        total_loss = (
            self.similarity_weight * sim_loss
            + self.variance_weight * var_loss
            + self.covariance_weight * cov_loss
        )
        # NaN/Inf 방지
        if not torch.isfinite(total_loss):
            total_loss = x.sum() * 0
        return total_loss

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
        vision_model_name: str = "google/siglip-base-patch16-224",
        language_model_name: str = "Qwen/Qwen3-0.6B",
        resampler_type: str = "mlp",
        latent_dimension: int = 768,
        vicreg_loss_weight: float = 1.0,
        vicreg_overlap_ratio: float = 0.5,
        max_text_length: int = 512,
    ):
        super().__init__()

        # 비전 인코더 초기화 ------------------------------------------------
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True)
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

        # VICReg 손실 함수 및 가중치 / 중첩 비율 ---------------------------
        self.vicreg_loss = VicRegLoss()
        self.vicreg_loss_weight = vicreg_loss_weight
        self.vicreg_overlap_ratio = vicreg_overlap_ratio
        
        # 텍스트 처리 관련 설정
        self.max_text_length = max_text_length
        self.ignore_index = -100
        # num_views 경고 플래그
        self._warned_single_view = False

    def _setup_tokenizer(self):
        """토크나이저 설정 개선"""
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # 특수 토큰 추가 (필요한 경우)
        special_tokens = {
            'additional_special_tokens': ['<|vision|>', '<|endoftext|>']
        }
        
        # 토크나이저에 특수 토큰이 없는 경우에만 추가
        if not any('<|vision|>' in str(token) for token in self.tokenizer.additional_special_tokens):
            self.tokenizer.add_special_tokens(special_tokens)
            # 언어 모델의 임베딩 크기도 조정
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        
        # 좌측 패딩 설정 (생성 시 필요)
        self.tokenizer.padding_side = "right"  # 학습 시는 right, 생성 시는 left로 변경

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
    
    def _create_combined_inputs(self, vision_tokens, text_inputs):
        """비전 토큰과 텍스트 입력을 결합 - 개선된 버전"""
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
        valid_text_labels = (text_inputs['labels'] != self.ignore_index).sum()
        # print(f"[AR Debug] Valid text labels: {valid_text_labels.item()}")
        # print(f"[AR Debug] Vision labels shape: {vision_labels.shape}")
        # print(f"[AR Debug] Text labels shape: {text_inputs['labels'].shape}")
        
        return {
            'inputs_embeds': combined_embeddings,
            'attention_mask': combined_attention,
            'labels': combined_labels
        }

    # ---------------- 메인 순전파 함수 -------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "vision", 
        **kwargs
    ):
        # 입력 차원 검증 및 reshape
        if pixel_values.ndim == 5:
            batch_size, num_views, channels, height, width = pixel_values.shape
        elif pixel_values.ndim == 4:
            batch_size, channels, height, width = pixel_values.shape
            num_views = 1
        else:
            raise ValueError(f"pixel_values shape invalid: {pixel_values.shape}")
        flattened_pixel_values = pixel_values.view(batch_size * num_views, channels, height, width)
        
        # 비전 특징 추출
        vision_output = self.vision_encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state  # (B*V, P, D)
        # LayerNorm 정규화 (VICReg 안정화)
        vision_hidden_states = self.vicreg_norm(vision_hidden_states)
        
        if stage == "vision":
            if num_views <= 1 and not self._warned_single_view:
                print("[VICReg] Warning: num_views <= 1, VICReg 손실이 0이 됩니다.")
                self._warned_single_view = True
            overlap_ratio = kwargs.get('overlap_ratio', self.vicreg_overlap_ratio)
            vicreg_raw = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, overlap_ratio=overlap_ratio
            )
            vicreg_loss = vicreg_raw * self.vicreg_loss_weight
            return {
                "loss": vicreg_loss,
                "vicreg_loss": vicreg_loss,          # train.py 로깅 호환
                "vicreg_raw": vicreg_raw.detach(),
                "vicreg_weight": self.vicreg_loss_weight
            }
        
        # 리샘플러 → 투영 후 AR 학습 단계
        resampled_features = self.resampler(vision_hidden_states)
        
        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            return self._compute_autoregressive_loss(
                resampled_features, input_ids, attention_mask, labels
            )
        else:
            raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    # ---------------- VICReg 중첩 영역 손실 계산 헬퍼 함수 ------------------------------------
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
        
        right = grid_features[..., -overlap_columns:, :]
        left = torch.roll(grid_features, shifts=-1, dims=1)[..., :overlap_columns, :]
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

    # ---------------- 자기회귀 손실 계산 함수 (개선된 버전) ------------------
    def _compute_autoregressive_loss(self, image_features, input_ids, attention_mask, labels):
        # 1. 비전 특징을 언어 모델 임베딩 공간으로 투영
        vision_tokens = self._project_vision_tokens(image_features)
        # 2. 텍스트 입력 전처리
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels)
        # 3. 비전과 텍스트 입력 결합
        combined_inputs = self._create_combined_inputs(vision_tokens, text_inputs)
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
                # manual dict에 ar_loss 키 보장
                manual.setdefault('ar_loss', manual['loss'])
                return manual
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

    # ---------------- 개선된 텍스트 생성 함수 -----------------------------
    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, input_ids: Optional[torch.Tensor] = None, 
                 max_new_tokens: int = 32, temperature: float = 0.7, **kwargs):
        """
        개선된 파노라마 이미지 텍스트 생성 함수
        
        Args:
            pixel_values: 파노라마 이미지 픽셀 값 (배치, 뷰수, 채널, 높이, 너비)
            input_ids: 입력 텍스트 토큰 ID (배치, 시퀀스길이) 또는 None
            max_new_tokens: 생성할 최대 새 토큰 수
            temperature: 샘플링 온도
        
        Returns:
            dict: {"generated_ids", "text"}
        """
        self.eval()
        
        # 입력 차원 검증 및 정규화
        batch_size, pixel_values = self._normalize_pixel_values(pixel_values)
        
        try:
            # 1. 비전 특징 추출 및 처리
            vision_tokens = self._extract_and_process_vision_features(pixel_values, batch_size)
            
            # 2. input_ids가 None이면 에러 (데이터셋에서 항상 제공되어야 함)
            if input_ids is None:
                raise ValueError("input_ids must be provided from dataset")
            
            # 어텐션 마스크가 없으면 생성
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            # 3. 생성 실행
            result = self._execute_generation(
                vision_tokens, input_ids, attention_mask, 
                max_new_tokens, temperature, **kwargs
            )
            
            return result
            
        except Exception as e:
            print(f"[Generate] Error in generation: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_generation_result(batch_size, pixel_values.device)
    
    def _normalize_pixel_values(self, pixel_values):
        """픽셀 값 차원 정규화"""
        if pixel_values.ndim == 5:
            batch_size, num_views, channels, height, width = pixel_values.shape
        elif pixel_values.ndim == 4:
            batch_size, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.unsqueeze(1)  # 뷰 차원 추가
        else:
            raise ValueError(f"pixel_values 차원이 잘못됨: {pixel_values.shape}")
        
        return batch_size, pixel_values
    
    def _extract_and_process_vision_features(self, pixel_values, batch_size):
        """비전 특징 추출 및 처리"""
        # 비전 인코더 처리를 위해 차원 변환
        num_views = pixel_values.size(1)
        flattened_pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        
        # 비전 인코더 통과
        vision_output = self.vision_encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state
        
        # 리샘플러 통과
        resampled_features = self.resampler(vision_hidden_states)
        
        # 배치 차원 복원
        seq_len = resampled_features.size(1)
        feature_dim = resampled_features.size(2)
        resampled_features = resampled_features.view(batch_size, num_views * seq_len, feature_dim)
        
        # 언어 모델 임베딩 공간으로 투영
        vision_tokens = self._project_vision_tokens(resampled_features)
        
        return vision_tokens
    
    def _execute_generation(self, vision_tokens, input_ids, attention_mask, 
                          max_new_tokens, temperature, **kwargs):
        """실제 생성 실행"""
        # input_ids 검증 (데이터셋에서 제공되어야 함)
        if input_ids is None:
            raise ValueError("input_ids must be provided from dataset")
            
        # 텍스트 임베딩 생성
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # 배치 크기 일치 확인
        batch_size = min(vision_tokens.size(0), text_embeddings.size(0))
        vision_tokens = vision_tokens[:batch_size]
        text_embeddings = text_embeddings[:batch_size]
        attention_mask = attention_mask[:batch_size]
        
        # 임베딩 결합
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
        
        # 어텐션 마스크 결합
        vision_mask = torch.ones(
            batch_size, vision_tokens.size(1), 
            dtype=torch.long, device=vision_tokens.device
        )
        combined_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        # 생성 파라미터 설정
        generation_kwargs = {
            'inputs_embeds': combined_embeddings,
            'attention_mask': combined_attention_mask,
            'max_new_tokens': max_new_tokens,
            'temperature': max(0.1, min(temperature, 1.0)),  # 온도 범위 제한
            'do_sample': True if temperature > 0.1 else False,
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 50),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.1),
            'length_penalty': kwargs.get('length_penalty', 1.0),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            # 'early_stopping': kwargs.get('early_stopping', True),
        }
        
        # 생성 실행
        generated_ids = self.language_model.generate(**generation_kwargs)
        
        # 텍스트 디코딩
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        # 프롬프트 제거 및 정리
        cleaned_texts = self._clean_generated_texts(generated_texts)
        
        return {
            "generated_ids": generated_ids,
            "text": cleaned_texts
        }
    
    def _clean_generated_texts(self, generated_texts, max_length: int = 200):
        """
        챗 템플릿이 적용된 생성 텍스트를 후처리하여 사용자 친화적으로 정리
        
        처리 과정:
        1. 챗 템플릿의 특수 토큰 및 구조 제거 (assistant 응답 부분만 추출)
        2. 빈 문자열에 대한 fallback 텍스트 제공
        3. 과도하게 긴 텍스트를 적절한 길이로 제한
        
        Args:
            generated_texts: 언어모델이 생성한 원시 텍스트 리스트
            max_length: 최대 허용 문자 수 (기본값: 200)
            
        Returns:
            list: 정리된 텍스트 리스트
        """
        cleaned_texts = []
        
        # 챗 템플릿의 assistant 응답 패턴들 (데이터셋에서 사용하는 패턴)
        assistant_patterns = [
            "<|im_start|>assistant\n",    # Qwen 챗 템플릿 (주로 사용)
            "<|im_start|>assistant",      # Qwen 개행 없음
            "### Assistant:\n",           # LLaMA/Vicuna
            "### Assistant:",
            "ASSISTANT:\n",               # 일반
            "ASSISTANT:",
            "<|assistant|>\n",            # ChatML
            "<|assistant|>",
            "### Response:\n",            # Alpaca
            "### Response:",
            "[/INST]\n",                  # Mistral
            "[/INST]",
            "Assistant:\n",               # 기본
            "Assistant:",
            "assistant:\n",               # 소문자
            "assistant:",
        ]
        
        # 템플릿 종료 토큰들
        end_tokens = [
            "<|im_end|>",                 # Qwen
            "<|endoftext|>",             # GPT
            "</s>",                      # LLaMA
            "[INST]",                    # Mistral 다음 턴
            "User:",                     # 다음 사용자 입력
            "### User:",
            "user:",
            "\nuser",                    # 개행 후 user
            "\nassistant",               # 개행 후 assistant
            "user\n",                    # user 후 개행
            "assistant\n",               # assistant 후 개행
        ]
        
        for text in generated_texts:
            cleaned_text = text.strip()
            
            # 디버깅: 원본 텍스트의 일부 확인
            if len(cleaned_texts) < 2:  # 처음 2개만 로그
                print(f"[DEBUG] Raw generated text #{len(cleaned_texts)}: '{cleaned_text[:100]}...'")
            
            # 1. Assistant 응답 부분만 추출
            assistant_content = None
            for pattern in assistant_patterns:
                if pattern in cleaned_text:
                    parts = cleaned_text.split(pattern, 1)
                    if len(parts) > 1:
                        assistant_content = parts[-1].strip()
                        if len(cleaned_texts) < 2:
                            print(f"[DEBUG] Found pattern '{pattern}', extracted: '{assistant_content[:50]}...'")
                        break
            
            # assistant 패턴을 찾지 못한 경우, 전체 텍스트를 사용
            if assistant_content is None:
                assistant_content = cleaned_text
                if len(cleaned_texts) < 2:
                    print(f"[DEBUG] No assistant pattern found, using full text")
            
            # 2. 종료 토큰 제거 (강화)
            for end_token in end_tokens:
                if end_token in assistant_content:
                    assistant_content = assistant_content.split(end_token)[0].strip()
                    if len(cleaned_texts) < 2:
                        print(f"[DEBUG] Removed end token '{end_token}'")
            
            # 3. 추가 정리: 대화 형식의 불필요한 부분 제거
            lines = assistant_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # 빈 줄이나 단순한 대화 키워드만 있는 줄 제거
                if line and not line.lower() in ['user', 'assistant', 'human', 'ai']:
                    # user: 또는 assistant: 로 시작하는 줄도 제거
                    if not (line.lower().startswith('user:') or 
                           line.lower().startswith('assistant:') or
                           line.lower().startswith('human:') or
                           line.lower().startswith('ai:')):
                        cleaned_lines.append(line)
            
            assistant_content = ' '.join(cleaned_lines).strip()
            
            # 4. 빈 문자열이나 너무 짧은 텍스트 처리
            if not assistant_content or len(assistant_content) < 3:
                assistant_content = "a panoramic scene"
                if len(cleaned_texts) < 2:
                    print(f"[DEBUG] Used fallback text")
            
            # 5. 길이 제한 (단어 단위로 자르기)
            if len(assistant_content) > max_length:
                # 단어 경계에서 자르기
                truncated = assistant_content[:max_length].rsplit(' ', 1)[0]
                # 마지막이 구두점이 아니면 ... 추가
                if truncated and not truncated[-1] in '.!?':
                    truncated += "..."
                assistant_content = truncated
                if len(cleaned_texts) < 2:
                    print(f"[DEBUG] Truncated to {len(assistant_content)} chars")
            
            # 6. 기본적인 문장 정리
            if assistant_content and assistant_content[0].islower():
                assistant_content = assistant_content[0].upper() + assistant_content[1:]
            
            # 7. 불필요한 반복 제거 (같은 문장이 반복되는 경우)
            sentences = assistant_content.split('. ')
            if len(sentences) > 1:
                # 연속된 같은 문장 제거
                unique_sentences = [sentences[0]]
                for sentence in sentences[1:]:
                    if sentence.strip() != unique_sentences[-1].strip():
                        unique_sentences.append(sentence)
                assistant_content = '. '.join(unique_sentences)
            
            # 최종 결과 디버깅
            if len(cleaned_texts) < 2:
                print(f"[DEBUG] Final cleaned text #{len(cleaned_texts)}: '{assistant_content}'")
            
            cleaned_texts.append(assistant_content)
        
        return cleaned_texts
    
    def _get_fallback_generation_result(self, batch_size, device):
        """Fallback 생성 결과"""
        fallback_text = ["a panoramic view"] * batch_size
        fallback_ids = torch.ones((batch_size, 5), dtype=torch.long, device=device)
        return {"generated_ids": fallback_ids, "text": fallback_text}

    def _project_vision_tokens(self, image_features):
        """비전 특징을 언어모델 임베딩 공간으로 투영"""
        return self.vision_to_language_projection(image_features)
    
    # ==================== LoRA 지원 메서드들 ====================
    
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
    
    def save_lora_weights(self, save_path: str):
        """LoRA 가중치만 저장"""
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot save LoRA weights.")
            return
        
        if hasattr(self.language_model, 'save_pretrained'):
            try:
                self.language_model.save_pretrained(save_path)
                print(f"✓ LoRA weights saved to: {save_path}")
            except Exception as e:
                print(f"Error saving LoRA weights: {e}")
        else:
            print("Warning: Language model does not support LoRA weight saving.")
    
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