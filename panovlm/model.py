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
            # print(f"[VICRegLoss] Warning: loss is not finite! sim: {sim_loss.item()} var: {var_loss.item()} cov: {cov_loss.item()}")
            total_loss = x.sum() * 0
        return total_loss

# ---------------------------------------------------------------------------
# ‣ PanoramaVLM (VICReg + AR)
# ---------------------------------------------------------------------------
class PanoramaVLM(nn.Module):
    def _mean_pool_vision_tokens(self, vision_hidden_states, batch_size, num_views, num_patches, vision_dimension):
        """
        vision_hidden_states: (batch*view, patch, dim)
        → (batch, view, patch, dim) → (batch, view, H, W, dim) → AdaptiveAvgPool2d(7x7) → (batch, view, 7, 7, dim)
        """
        reshaped_features = vision_hidden_states.view(batch_size, num_views, num_patches, vision_dimension)
        grid_height, grid_width = _infer_hw(num_patches)
        grid_features = reshaped_features.view(batch_size, num_views, grid_height, grid_width, vision_dimension)
        # print(f"[VLM] Grid shape: {grid_features.shape} (batch, view, H, W, dim)")
        # (batch, view, H, W, dim) → (batch*view, dim, H, W)
        x = grid_features.permute(0,1,4,2,3).contiguous()  # (batch, view, dim, H, W)
        x = x.view(batch_size * num_views, vision_dimension, grid_height, grid_width)
        pooled = F.adaptive_avg_pool2d(x, (grid_height // 2, grid_width // 2))  # (batch*view, dim, H//2, W//2)
        # (batch*view, dim, H//2, W//2) → (batch, view, H//2, W//2, dim)
        pooled = pooled.view(batch_size, num_views, vision_dimension, grid_height // 2, grid_width // 2)
        pooled = pooled.permute(0, 1, 3, 4, 2).contiguous()  # (batch, view, H//2, W//2, dim)
        # print(f"[VLM] Pooled features shape: {pooled.shape} (batch, view, {grid_height//2}, {grid_width//2}, dim)")
        # (batch, view, H//2, W//2, dim) → (batch, view * H//2 * W//2, dim)
        flattened = pooled.view(batch_size, num_views * (grid_height // 2) * (grid_width // 2), vision_dimension)
        # print(f"[VLM] Flattened pooled shape: {flattened.shape} (batch, seq, dim)")
        return flattened
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

        # 풀링 레이어 정의 ---------------------------------------------
        self.pooling = nn.AdaptiveAvgPool2d
        
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
        stage: str = "vision", 
        **kwargs
    ):
        """파노라마 VLM 순전파
        
        Args:
            pixel_values: 파노라마 이미지 픽셀 값 (배치, 뷰수, 채널, 높이, 너비)
            input_ids: 입력 텍스트 토큰 ID (배치, 시퀀스길이)
            attention_mask: 어텐션 마스크 (배치, 시퀀스길이)
            labels: 레이블 (배치, 시퀀스길이)
            stage: 학습 단계 ('vision', 'resampler', 'finetune')
            max_new_tokens: 생성 시 최대 새 토큰 수
            temperature: 생성 시 온도 파라미터
        
        Returns:
            dict: 손실 또는 생성 결과를 포함한 딕셔너리
        """
        # 입력 차원 검증
        if pixel_values.ndim == 5:
            batch_size, num_views, channels, height, width = pixel_values.shape
        elif pixel_values.ndim == 4:
            # (배치, 채널, 높이, 너비) 형태로 가정
            batch_size, channels, height, width = pixel_values.shape
            num_views = 1
        # 배치와 뷰 차원을 합쳐서 비전 인코더에 입력 가능한 형태로 변환
        pixel_values = pixel_values.view(batch_size * num_views, channels, height, width)
        # # print(f"비전 인코더 입력 형태: {flattened_pixel_values.shape}")
        
        # 비전 인코더를 통한 특징 추출 -----------------------------------
        vision_output = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state  # (배치*뷰수, 패치수, 비전차원)
        
        # # print(f"비전 인코더 출력 형태: {vision_hidden_states.shape}")
        _, num_patches, vision_dimension = vision_hidden_states.shape
        
        # 단계 1: VICReg 손실 계산 (비전 인코더만 학습) ------------------
        if stage == "vision":
            vicreg_loss = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, overlap_ratio=0.5
            )
            return {"loss": vicreg_loss}
        
        # 리샘플러를 통한 특징 변환
        resampled_features = self.resampler(vision_hidden_states)
        
        # 단계 2: 리샘플러 학습 (비전 인코더 + 리샘플러 학습) ---------------
        if stage == "resampler":
            # VICReg 손실과 autoregressive 손실을 모두 계산
            vicreg_loss = self._compute_vicreg_overlap_loss(
                vision_hidden_states, batch_size, num_views, overlap_ratio=0.5
            )
            ar_loss_output = self._compute_autoregressive_loss(resampled_features, input_ids, attention_mask, labels)
            
            # 가중치를 적용하여 두 손실을 결합
            total_loss = self.vicreg_loss_weight * vicreg_loss + ar_loss_output["loss"]
            return {"loss": total_loss, "vicreg_loss": vicreg_loss, "ar_loss": ar_loss_output["loss"]}
        
        # 단계 3: 전체 모델 파인튜닝 ----------------------------------------
        elif stage == "finetune":
            return self._compute_autoregressive_loss(resampled_features, input_ids, attention_mask, labels)
        
        else:
            raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다")

    # ---------------- VICReg 중첩 영역 손실 계산 헬퍼 함수 ------------------------------------
    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,   # (배치*뷰수, 패치수, 차원)
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,   # 파노라마 뷰 간 중첩 비율(열 기준)
    ):
        # 정규화 전 평균/분산 출력
        # # print("[VICReg] 정규화 전 mean:", vision_output.mean().item(), "var:", vision_output.var().item())
        # BatchNorm1d 적용: (배치*뷰수, 패치수, 차원) -> (배치*뷰수*패치수, 차원)
        flat = vision_output.reshape(-1, vision_output.shape[-1])
        bn = nn.BatchNorm1d(flat.shape[1], affine=True, eps=1e-5).to(flat.device)
        normed = bn(flat)
        vision_hidden_states = normed.view_as(vision_output)
        # 정규화 후 평균/분산 출력
        # # print("[VICReg] 정규화 후 mean:", vision_hidden_states.mean().item(), "var:", vision_hidden_states.var().item())
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
    def _compute_autoregressive_loss(
        self,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_model: AutoModelForCausalLM,
        pad_token_id: int = -100,
    ):
        """
        Args:
            image_features: (B, V*P, D₁) — Resampler를 거친 이미지 특징
            input_ids:       (B, T)       — 텍스트 입력 토큰
            attention_mask:  (B, T)       — 텍스트 어텐션 마스크 (1 for real tokens, 0 for pad)
            language_model:  AutoModelForCausalLM — causal LM
            pad_token_id:    int — CrossEntropy ignore_index (기본 -100)
        Returns:
            loss: torch.Tensor — 스칼라 CE loss
        """
        batch_size = input_ids.size(0)

        # 1) 이미지 임베딩 투영 → vision_tokens: (B, N_vis, D₂)
        vision_tokens = self.vision_to_language_projection(image_features)

        # 2) 텍스트 임베딩 생성 → text_embeds: (B, T, D₂)
        text_embeds = language_model.get_input_embeddings()(input_ids)

        # 3) combined embeddings → (B, N_vis + T, D₂)
        combined_embeddings = torch.cat([vision_tokens, text_embeds], dim=1)

        # 4) combined attention mask → (B, N_vis + T)
        vision_mask = torch.ones(batch_size, vision_tokens.size(1), device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # 5) labels 생성 (vision 토큰은 ignore, 텍스트는 shift + pad_token_id 처리)
        #    - vision_labels: (B, N_vis) filled with pad_token_id(-100)
        vision_labels = torch.full((batch_size, vision_tokens.size(1)), pad_token_id, device=input_ids.device, dtype=input_ids.dtype)

        #    - text_labels: (B, T) next-token 예측용 shift
        text_labels = input_ids.clone()
        text_labels[:, :-1] = input_ids[:, 1:]
        text_labels[:, -1] = pad_token_id

        combined_labels = torch.cat([vision_labels, text_labels], dim=1)

        
        # 7) 외부 라이브러리 스타일 안정적인 Loss 계산
        try:
            # Method 1: Transformers 라이브러리 내장 loss 사용 (가장 안정적)
            outputs = self.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_mask,
                labels=combined_labels,
                return_dict=True
            )
            
            # Transformers가 자동으로 올바른 shift와 loss 계산 수행
            loss = outputs.loss
            logits = outputs.logits
            
            # Loss 유효성 검사
            if not torch.isfinite(loss):
                print("[AR Loss] Warning: Non-finite loss detected, using fallback method")
                raise ValueError("Non-finite loss")
                
        except Exception as e:
            # Method 2: 수동 계산 (fallback)
            print(f"[AR Loss] Using manual calculation: {e}")
            
            # 언어 모델 순전파 (레이블 없이)
            outputs = self.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_mask,
                return_dict=True
            )
            logits = outputs.logits
            
            # 안정적인 Cross Entropy Loss 계산
            loss = self._compute_stable_cross_entropy_loss(logits, combined_labels)
        
        return {"loss": loss, "logits": logits}
    
    def _compute_stable_cross_entropy_loss(self, logits, labels, label_smoothing=0.0):
        """
        외부 라이브러리 스타일 안정적인 Cross Entropy Loss 계산
        Transformers 라이브러리와 동일한 방식
        """
        # Next token prediction을 위한 shift (Transformers 표준)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss calculation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Transformers 스타일 안정적인 Cross Entropy 계산
        loss = F.cross_entropy(
            shift_logits, 
            shift_labels,
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        
        # NaN/Inf 방지를 위한 추가 안전장치
        if not torch.isfinite(loss):
            print("[AR Loss] Warning: Non-finite loss in manual calculation")
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss

    # ---------------- 텍스트 생성 함수 ---------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_text(self, vision_tokens, input_ids=None, max_new_tokens: int = 32, temperature: float = 0.7):
        """비전 토큰을 이용한 텍스트 생성"""
        # vision_tokens: (batch, seq, dim)
        if input_ids is not None:
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)
            
            # 배치 크기 확인 및 조정
            vision_batch_size = vision_tokens.size(0)
            text_batch_size = text_embeddings.size(0)
            
            # 배치 크기가 다를 경우 맞춤
            if vision_batch_size != text_batch_size:
                min_batch_size = min(vision_batch_size, text_batch_size)
                vision_tokens = vision_tokens[:min_batch_size]
                text_embeddings = text_embeddings[:min_batch_size]
            
            combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)
            vision_attention_mask = torch.ones(vision_tokens.shape[:2], dtype=torch.long, device=vision_tokens.device)
            text_attention_mask = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=text_embeddings.device)
            attention_mask = torch.cat([vision_attention_mask, text_attention_mask], dim=1)
        else:
            combined_embeddings = vision_tokens
            attention_mask = torch.ones(combined_embeddings.shape[:2], dtype=torch.long, device=vision_tokens.device)

        # 생성 파라미터 개선
        generated_token_ids = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=min(temperature, 1.0),  # 온도 제한
            do_sample=True,
            top_p=0.9,  # nucleus sampling 추가
            top_k=50,   # top-k sampling 추가
            repetition_penalty=1.1,  # 반복 방지
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        return {"generated_ids": generated_token_ids, "text": generated_texts}

    def generate(self, pixel_values: torch.Tensor, input_ids: Optional[torch.Tensor] = None, 
                 max_new_tokens: int = 32, temperature: float = 0.7):
        """
        파노라마 이미지에서 텍스트를 생성하는 독립적인 함수
        
        Args:
            pixel_values: 파노라마 이미지 픽셀 값 (배치, 뷰수, 채널, 높이, 너비)
            input_ids: 입력 텍스트 토큰 ID (배치, 시퀀스길이) 또는 None
            max_new_tokens: 생성할 최대 새 토큰 수
            temperature: 샘플링 온도
        
        Returns:
            dict: {"generated_ids", "text"}
        """
        # 입력 차원 검증
        if pixel_values.ndim == 5:
            batch_size, num_views, channels, height, width = pixel_values.shape
        elif pixel_values.ndim == 4:
            # (배치, 채널, 높이, 너비) 형태로 가정
            batch_size, channels, height, width = pixel_values.shape
            num_views = 1
        
        # 배치와 뷰 차원을 합쳐서 비전 인코더에 입력 가능한 형태로 변환
        flattened_pixel_values = pixel_values.view(batch_size * num_views, channels, height, width)
        
        # 비전 인코더를 통한 특징 추출
        vision_output = self.vision_encoder(pixel_values=flattened_pixel_values, return_dict=True)
        vision_hidden_states = vision_output.last_hidden_state  # (배치*뷰수, 패치수, 비전차원)
        
        # 리샘플러를 통한 특징 변환
        resampled_features = self.resampler(vision_hidden_states)
        
        # 비전 특징을 언어 모델 임베딩 공간으로 투영
        vision_tokens = self._project_vision_tokens(resampled_features)
        
        # 텍스트 생성
        return self._generate_text(vision_tokens, input_ids=input_ids, 
                                 max_new_tokens=max_new_tokens, temperature=temperature)

    def _project_vision_tokens(self, image_features):
        """비전 특징을 언어모델 임베딩 공간으로 투영"""
        return self.vision_to_language_projection(image_features)