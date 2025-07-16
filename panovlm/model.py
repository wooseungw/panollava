# coding: utf-8
"""
PanoramaVLM – 2-Stage (VICReg ➜ AR) 평가용 레퍼런스 구현
────────────────────────────────────────────────────────
● **Stage "vicreg"** : 파노라마 인접 뷰 간 중첩 일관성 정렬(VICReg loss)
● **Stage "train"**  : Vision → Resampler → LLM 단일 autoregressive CE loss
● **Stage "generate"** : 프롬프트 없는 zero-shot 캡션/응답 생성

본 파일은 **데이터셋-호환 & 그래디언트 검증용 테스트 코드**를 포함합니다.
(데이터셋이 dict 형태로
  `{pixel_values, input_ids, attention_mask, labels}`
  를 반환한다고 가정)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
        # # 입력값 클리핑
        # x = torch.clamp(x, min=-10, max=10)
        # y = torch.clamp(y, min=-10, max=10)

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
            print(f"[VICRegLoss] Warning: loss is not finite! sim: {sim_loss.item()} var: {var_loss.item()} cov: {cov_loss.item()}")
            total_loss = x.sum() * 0
        return total_loss

# ---------------------------------------------------------------------------
# ‣ PanoramaVLM (VICReg + AR)
# ---------------------------------------------------------------------------
class PanoramaVLM(nn.Module):
    """파노라마 비전-언어 모델
    
    2단계 학습 분기 구조:
    1) VICReg: 파노라마 뷰 간 중첩 영역 일관성 학습
    2) Autoregressive: 비전-텍스트 생성 학습
    """
    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-224",
        language_model_name: str = "Qwen/Qwen3-0.6B",
        resampler_type: str = "mlp",
        latent_dimension: int = 768,
        vicreg_loss_weight: float = 1.0,
    ):
        super().__init__()

        # 비전 인코더 초기화 ------------------------------------------------
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True)
        if hasattr(self.vision_encoder, "vision_model"):
            self.vision_encoder = self.vision_encoder.vision_model
        vision_hidden_size = self._get_vision_hidden_size(self.vision_encoder)

        # 리샘플러 초기화 ---------------------------------------------------
        if resampler_type == "mlp":
            self.resampler = MLPResampler(vision_hidden_size, latent_dimension)
        else:
            raise ValueError(f"지원하지 않는 리샘플러 타입: {resampler_type}")

        # 언어 모델 초기화 ---------------------------------------------------
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        # 비전 특징을 언어 모델 임베딩 공간으로 투영하는 레이어
        self.vision_to_language_projection = nn.Linear(latent_dimension, self.language_model.config.hidden_size)

        # 토크나이저 초기화 -------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # VICReg 손실 함수 초기화 -------------------------------------------
        self.vicreg_loss = VicRegLoss()
        self.vicreg_loss_weight = vicreg_loss_weight

    # ---------------- 유틸리티 함수 ---------------------------------------------
    @staticmethod
    def _get_vision_hidden_size(vision_model: nn.Module) -> int:
        """비전 모델의 은닉 차원 크기를 추출"""
        possible_keys = ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]
        for key in possible_keys:
            if hasattr(vision_model.config, key):
                return getattr(vision_model.config, key)
        raise AttributeError("비전 모델의 은닉 차원 크기를 찾을 수 없습니다")

    # ---------------- 메인 순전파 함수 -------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "train",  # "vision" | "finetune" | "generate"
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        **kwargs
    ):
        """파노라마 VLM 순전파
        
        Args:
            pixel_values: 파노라마 이미지 픽셀 값 (배치, 뷰수, 채널, 높이, 너비)
            input_ids: 입력 텍스트 토큰 ID (배치, 시퀀스길이)
            attention_mask: 어텐션 마스크 (배치, 시퀀스길이)
            labels: 레이블 (배치, 시퀀스길이)
            stage: 학습 단계 ("vision", "finetune", "generate")
            max_new_tokens: 생성 시 최대 새 토큰 수
            temperature: 생성 시 온도 파라미터
        
        Returns:
            dict: 손실 또는 생성 결과를 포함한 딕셔너리
        """
        # 입력 차원 검증
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values는 (배치, 뷰수, 채널, 높이, 너비) 형태여야 합니다")
        
        batch_size, num_views, channels, height, width = pixel_values.shape
        # 배치와 뷰 차원을 합쳐서 비전 인코더에 입력 가능한 형태로 변환
        flattened_pixel_values = pixel_values.view(batch_size * num_views, channels, height, width)
        # print(f"비전 인코더 입력 형태: {flattened_pixel_values.shape}")
        
        # 비전 인코더를 통한 특징 추출 -----------------------------------
        vision_output = self.vision_encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state  # (배치*뷰수, 패치수, 비전차원)
        
        # print(f"비전 인코더 출력 형태: {vision_hidden_states.shape}")
        _, num_patches, vision_dimension = vision_hidden_states.shape
        
        # 단계 1: VICReg 손실 계산 (비전 인코더만 학습) ------------------
        if stage == "vision":
            vicreg_loss = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, overlap_ratio=0.5
            )
            return {"loss": vicreg_loss}

        # 리샘플러를 통한 특징 변환 -------------------------------------
        # (배치*뷰수, 패치수, 비전차원) → (배치, 뷰수*패치수, 비전차원) → (배치, 뷰수*패치수, 잠재차원)
        reshaped_features = vision_hidden_states.view(batch_size, num_views, num_patches, vision_dimension)
        flattened_features = reshaped_features.reshape(batch_size, num_views * num_patches, vision_dimension)
        resampled_features = self.resampler(flattened_features)
        
        # 단계 2: 텍스트 생성 ----------------------------------------------
        if stage == "generate":
            return self._generate_text(resampled_features, max_new_tokens, temperature)
        elif stage == "finetune":
            return self._compute_autoregressive_loss(resampled_features, input_ids, attention_mask, labels)
        else:
            raise ValueError("stage는 'vision', 'finetune', 또는 'generate' 중 하나여야 합니다")

    # ---------------- VICReg 중첩 영역 손실 계산 헬퍼 함수 ------------------------------------
    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,   # (배치*뷰수, 패치수, 차원)
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,   # 파노라마 뷰 간 중첩 비율(열 기준)
    ):
        # 정규화 전 평균/분산 출력
        # print("[VICReg] 정규화 전 mean:", vision_output.mean().item(), "var:", vision_output.var().item())
        # BatchNorm1d 적용: (배치*뷰수, 패치수, 차원) -> (배치*뷰수*패치수, 차원)
        flat = vision_output.reshape(-1, vision_output.shape[-1])
        bn = nn.BatchNorm1d(flat.shape[1], affine=True, eps=1e-5).to(flat.device)
        normed = bn(flat)
        vision_hidden_states = normed.view_as(vision_output)
        # 정규화 후 평균/분산 출력
        # print("[VICReg] 정규화 후 mean:", vision_hidden_states.mean().item(), "var:", vision_hidden_states.var().item())
        """배치 내 모든 인접 뷰 쌍, 모든 위치(높이, 오버랩 열)별 오버랩 패치 쌍을 벡터화하여 VICReg loss 평균 계산"""
        if num_views <= 1:
            return torch.zeros((), device=vision_hidden_states.device)

        # 1) CLS 토큰 유무 판단 및 제거
        has_cls_token = (vision_hidden_states.shape[1] % 2 == 1)
        patch_features = vision_hidden_states[:, 1:] if has_cls_token else vision_hidden_states
        num_patches = patch_features.size(1)

        # 2) 패치를 2D 그리드로 복원
        grid_height, grid_width = _infer_hw(num_patches)
        grid_features = patch_features.view(batch_size, num_views, grid_height, grid_width, -1)

        # 3) 중첩 영역 크기 계산
        overlap_columns = max(1, int(grid_width * overlap_ratio))

        # 4) 오른쪽 k개 열 (현재 뷰)
        right = grid_features[..., -overlap_columns:, :]  # (B, V, H, k, C)
        # 5) 왼쪽 k개 열 (다음 뷰)
        left = torch.roll(grid_features, shifts=-1, dims=1)[..., :overlap_columns, :]  # (B, V, H, k, C)

        # 6) (B, V, H, k, C) → (N, C)로 펼치기
        right_flat = right.reshape(-1, right.shape[-1])
        left_flat = left.reshape(-1, left.shape[-1])

        # 7) VICRegLoss에 한 번에 전달
        if right_flat.shape[0] == 0:
            return torch.zeros((), device=vision_hidden_states.device)
        return self.vicreg_loss(right_flat, left_flat)

    # ---------------- 자기회귀 손실 계산 함수 ------------------------------------------
    def _compute_autoregressive_loss(self, image_features, input_ids, attention_mask, labels):
        """비전-언어 모델의 자기회귀 손실 계산
        
        Args:
            image_features: 이미지 특징 (배치, 뷰수*패치수, 차원)
            input_ids: 텍스트 토큰 ID (배치, 텍스트길이)
            attention_mask: 텍스트 어텐션 마스크 (배치, 텍스트길이)
            labels: 다음 토큰 예측을 위한 레이블 (배치, 텍스트길이)
        
        Returns:
            dict: 손실과 로짓을 포함한 딕셔너리
        """
        # 1) 이미지 특징을 언어 모델 임베딩 공간으로 투영
        vision_tokens = self.vision_to_language_projection(image_features)  # (배치, 뷰수*패치수, 언어모델_차원)
        
        # 2) 텍스트 토큰을 임베딩으로 변환
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)  # (배치, 텍스트길이, 언어모델_차원)
        
        # 3) 비전 토큰과 텍스트 임베딩을 시퀀스 차원에서 연결
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)  # (배치, 비전+텍스트길이, 언어모델_차원)
        
        # 4) 어텐션 마스크 생성: 비전 토큰은 항상 어텐션 받음, 텍스트는 주어진 마스크 사용
        vision_attention_mask = torch.ones(vision_tokens.shape[:2], dtype=torch.long, device=vision_tokens.device)
        combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)  # (배치, 비전+텍스트길이)
        
        # 5) 레이블 생성: 비전 토큰 부분은 손실 계산에서 제외(-100), 텍스트 부분은 주어진 레이블 사용
        vision_labels_ignore = torch.full((vision_tokens.size(0), vision_tokens.size(1)), -100, 
                                         dtype=torch.long, device=vision_tokens.device)
        combined_labels = torch.cat([vision_labels_ignore, labels], dim=1)  # (배치, 비전+텍스트길이)
        
        # 6) 언어 모델 순전파
        language_model_output = self.language_model(
            inputs_embeds=combined_embeddings, 
            attention_mask=combined_attention_mask, 
            labels=combined_labels
        )
        
        return {"loss": language_model_output.loss, "logits": language_model_output.logits}

    # ---------------- 텍스트 생성 함수 ---------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_text(self, image_features, max_new_tokens: int, temperature: float):
        """이미지 특징을 기반으로 텍스트 생성
        
        Args:
            image_features: 이미지 특징 (배치, 뷰수*패치수, 차원)
            max_new_tokens: 생성할 최대 새 토큰 수
            temperature: 샘플링 온도 (높을수록 더 다양한 생성)
        
        Returns:
            dict: 생성된 토큰 ID와 텍스트를 포함한 딕셔너리
        """
        # 1) 이미지 특징을 언어 모델 임베딩 공간으로 투영
        vision_tokens = self.vision_to_language_projection(image_features)  # (배치, 뷰수*패치수, 언어모델_차원)
        
        # 2) 시작 토큰 준비 (BOS 토큰 또는 EOS 토큰 사용)
        start_token_id = self.language_model.config.bos_token_id or self.tokenizer.eos_token_id
        start_token_tensor = torch.tensor([[start_token_id]], device=vision_tokens.device)
        start_token_embedding = self.language_model.get_input_embeddings()(start_token_tensor)
        
        # 3) 비전 토큰과 시작 토큰을 연결
        # 배치 크기에 맞춰 시작 토큰 확장
        expanded_start_embedding = start_token_embedding.expand(vision_tokens.size(0), -1, -1)
        combined_embeddings = torch.cat([vision_tokens, expanded_start_embedding], dim=1)
        
        # 4) 어텐션 마스크 생성 (모든 토큰에 어텐션 적용)
        attention_mask = torch.ones(combined_embeddings.shape[:2], dtype=torch.long, device=vision_tokens.device)

        # 5) 언어 모델을 사용하여 텍스트 생성
        generated_token_ids = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,  # 샘플링 활성화
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # 6) 생성된 토큰을 텍스트로 디코딩
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        
        return {"generated_ids": generated_token_ids, "text": generated_texts}

