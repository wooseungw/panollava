
## PanoVLM — 3단계 학습 흐름 (레이어/모듈 관점)

이 문서는 PanoVLM의 Progressive Training(3단계)의 목적, 데이터 흐름, 손실 계산, 그리고 각 단계에서 학습되는 파라미터와 그래디언트 흐름을 간결하게 정리합니다.

---

### 1단계: Vision Stage (VICReg — Self-Supervised)

목적
- Resampler가 multi-view 파노라마 간의 공간적 일관성(spatial consistency)을 학습합니다.

입력/출력 요약
- Input: `pixel_values` — [B, V, 3, H, W] (multi-view)
- Vision Encoder (frozen): 출력 → [B*V, S, D_vision]
- Resampler (trainable): 입력 [B*V, S, D_vision] → 출력 [B*V, S', D_latent]
- VICReg Projector (trainable): [B*V, S', D_latent] → [B*V, S', D_vicreg]

VICReg Overlap Loss (요약)
1. Reshape: [B*V, S', D] → [B, V, H', W', D] (H', W' ≈ sqrt(S'))
2. 겹치는 영역 추출:
   - curr = view v의 오른쪽 k 열
   - next = view v+1의 왼쪽 k 열
   - k = int(W' * overlap_ratio)
3. Flatten: [B, V, H', k, D] → [P, L, D], P=B×V, L=H'×k
4. VICReg 구성요소:
   - Invariance: MSE(curr, next)
   - Variance: relu(γ - std(features)) (collapse 방지)
   - Covariance: off-diagonal(cov)^2 (feature decorrelation)
5. 총손실: loss = 25×inv + 25×var + 1×cov

그래디언트 흐름
- VICReg Projector ← train
- Resampler ← train
- Vision Encoder: frozen (gradient 차단)

학습되는 파라미터
- Resampler: ✅
- VICReg Projector: ✅
- Vision Encoder: ❄️ (동결)

---

### 2단계: Resampler Stage (Vision → Language Alignment)

목적
- Resampler 출력을 언어모델(LM) 공간으로 정렬하여, vision token이 LM에서 이해될 수 있도록 합니다.

입력/출력 요약
- Input: `pixel_values` [B, V, 3, H, W], `input_ids` [B, L], `labels` [B, L]
- Vision Encoder: frozen → [B*V, S, D_vision]
- Resampler: trainable (1단계에서 학습된 가중치 사용) → [B*V, S', D_latent]
- Multi-view pooling (옵션): mean 또는 concat → [B, S'_total, D_latent]
- Projection Layer (trainable): Linear D_latent → D_lm → `vision_tokens` [B, S'_total, D_lm]
- LM Embedding: `text_embeds` [B, L_text, D_lm]
- Combine: concat(vision_tokens, text_embeds) → LM에 입력

Loss 및 학습
- Loss: Autoregressive CrossEntropy (text token만 계산; vision token 위치는 labels=-100으로 마스킹)
- LM 본체는 frozen(embedding은 forward만), Projection Layer 및 Resampler는 학습됩니다.

그래디언트 흐름
- Projection Layer ← train
- Resampler ← train
- Language Model (base) ✖️ (동결)
- Vision Encoder ✖️ (동결)

학습되는 파라미터
- Resampler: ✅
- Projection Layer: ✅
- LM (base): ❄️ (동결)
- VICReg Projector: ❌ (1단계 전용)

---

### 3단계: Finetune Stage (Instruction Tuning — End-to-End)

목적
- 전체 파이프라인을 instruction-following task에 맞게 미세조정합니다. 보통 LM 본체는 동결하고 LoRA와 같은 경량 어댑터만 학습합니다.

입력/출력 요약
- Input: `pixel_values` [B, V, 3, H, W], `input_ids` [B, L]
- Labels: response 부분만 loss 계산, prompt와 vision token은 `-100`으로 마스킹
- Resampler, Projection Layer: trainable (과거 단계 가중치로 초기화)
- Language Model: base weights frozen, LoRA adapters (optional) trainable

Loss 및 학습
- Loss: Autoregressive CrossEntropy (response token만 계산)

그래디언트 흐름
- Resampler ← train
- Projection Layer ← train
- Language Model LoRA ← train (선택적)
- Vision Encoder: 기본 동결, 필요 시 마지막 N개 레이어만 unfreeze 가능

학습되는 파라미터
- Resampler: ✅
- Projection Layer: ✅
- LM (LoRA): ✅ (옵션)
- Vision Encoder: ❄️ 또는 부분 학습

---

## 손실 크기(대략적)
- 1단계 (VICReg): 초기 50~100 → 수렴 10~30
- 2단계 (Alignment): 초기 8~12 (perplexity 높음) → 수렴 2~4
- 3단계 (Finetune): 초기 2~4 → 수렴 0.5~1.5

## 그래디언트 흐름 요약

Stage      | Vision Encoder | Resampler | VICReg Projector | Projection Layer | LM (Base) | LM (LoRA)
:---------:|:--------------:|:---------:|:-----------------:|:----------------:|:---------:|:-----------:
Stage 1    | ❄️ (frozen)    | ✅        | ✅                | ❌               | ❌        | ❌
Stage 2    | ❄️ (frozen)    | ✅        | ❌                | ✅               | ❄️        | ❌
Stage 3    | ❄️ (partial)   | ✅        | ❌                | ✅               | ❄️        | ✅ (optional)

## 핵심 요약
- 1단계: Resampler의 spatial consistency 학습 (언어 없음)
- 2단계: Vision → Language 정렬 (Projection layer 학습)
- 3단계: Instruction-tuning (LoRA로 LM 적응, 나머지 모듈 미세조정)
- Progressive Training 장점: 단계별 명확한 목적, 안정적 수렴, 효율적인 파라미터 업데이트

---

작성: repository 문서 형식에 맞게 간결화 및 구조화됨. 필요하면 표/그림으로 추가 확장하거나 각 단계의 하이퍼파라미터 예시(예: S', D_latent, D_lm, overlap_ratio)를 덧붙여 드리겠습니다.
