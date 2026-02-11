# default.yaml 기반 PanoLLaVA vs. 상용 VLM 설정 비교

## 📋 현재 설정 요약 (default.yaml)

```yaml
# 핵심 설정
vision_encoder: siglip2-so400m-patch16-256 (400M 파라미터)
language_model: Qwen3-0.6B (600M 파라미터)
resampler: BiMamba
crop_strategy: anyres_e2p (파노라마 특화)
positional_encoding: enabled
stages: [vision, resampler, finetune]
lora: enabled (rank=32, alpha=64)
```

---

## ✅ 공통점 (상용 VLM과 동일한 설계 선택)

### 1. **Vision Encoder 선택**

**PanoLLaVA (default.yaml)**:
```yaml
vision_name: "google/siglip2-so400m-patch16-256"
```

**상용 VLM**:
- ✅ LLaVA-OneVision: rice-vit (SigLIP 기반)
- ✅ Qwen2.5-VL: SigLIP-Large
- ✅ InternVL: InternViT (DINOv2 유사)

**공통점**: 모두 **CLIP/SigLIP 계열** 사용 (강력한 vision-language 정렬)

---

### 2. **Language Model 크기**

**PanoLLaVA (default.yaml)**:
```yaml
language_model_name: "Qwen/Qwen3-0.6B"  # 0.6B
```

**상용 VLM**:
- ✅ LLaVA-OneVision-0.5B: 0.5B (유사)
- ✅ Qwen2.5-VL: 2B, 7B, 32B
- ✅ InternVL: 1B, 2B, 4B, 8B

**공통점**: **소형 모델** 지원 (모바일/엣지 배포 고려)

---

### 3. **LoRA 활용**

**PanoLLaVA (default.yaml)**:
```yaml
lora:
  use_lora: true
  rank: 32  
  alpha: 64
  dropout: 0.1
  target_modules: ["q_proj", "k_proj"]
```

**상용 VLM**:
- ⚠️ 대부분 **LoRA를 공식적으로 지원하지 않음** (full fine-tuning)
- 일부 커뮤니티에서 LoRA 적용 시도

**부분 공통점**: LoRA는 **PEFT (Parameter-Efficient Fine-Tuning)** 트렌드를 따름

---

### 4. **Multi-Stage Training**

**PanoLLaVA (default.yaml)**:
```yaml
stages: ["vision", "resampler", "finetune"]
```

**상용 VLM**:
- ✅ LLaVA 계열: 2-3단계 (pretraining → SFT)
- ✅ Qwen2.5-VL: Vision pretraining → SFT
- ✅ InternVL: Multi-stage progressive training

**공통점**: **점진적 학습** (progressive training) 전략 사용

---

### 5. **Flash Attention 2 (암묵적)**

**PanoLLaVA**:
- 코드에서 Flash Attention 2 자동 감지 및 사용
- 메모리 ~30% 절감, 속도 ~2배 향상

**상용 VLM**:
- ✅ 모든 최신 VLM이 Flash Attention 사용

**공통점**: **메모리 최적화** 기법 표준화

---

## ❌ 차이점 (PanoLLaVA만의 독특한 설정)

### 1. **파노라마 특화 이미지 처리** ⭐⭐⭐

**PanoLLaVA (default.yaml)**:
```yaml
image_processing:
  crop_strategy: "anyres_e2p"  # Equirectangular-to-Perspective
  overlap_ratio: 0.5           # 50% 겹침
  fov_deg: 90.0                # 90도 시야각
  anyres_max_patches: 9        # 최대 9개 타일
```

**의미**:
- 360° 파노라마를 **여러 perspective 뷰로 변환**
- 인접 뷰 간 **50% 겹침** → VICReg loss에서 활용
- 최대 9개 타일 생성 → 고해상도 정보 보존

**상용 VLM**:
```yaml
# 일반적인 처리
image_processing:
  resize: true
  max_size: [384, 384]  # 또는 더 큼
  strategy: "letterbox"  # 또는 "center_crop"
```

**차이점**: 
- ❌ 상용 VLM: 단순 **리사이징/크롭**
- ✅ PanoLLaVA: **기하학적 변환 + 다중 뷰 생성**

---

### 2. **VICReg Loss (Overlap 정규화)** ⭐⭐⭐

**PanoLLaVA (default.yaml)**:
```yaml
vision:
  vicreg_loss_weight: 1.0
  vicreg_mode: "pairwise"  # 인접 뷰 쌍 비교
  vicreg_similarity_weight: 25.0   # Invariance
  vicreg_variance_weight: 25.0     # Variance
  vicreg_covariance_weight: 1.0    # Covariance
```

**의미**:
- **인접 뷰의 겹치는 영역**에서 feature 일관성 강제
- `25*invariance + 25*variance + 1*covariance`
- 파노라마의 **공간 연속성** 학습

**상용 VLM**:
```yaml
# VICReg loss 없음
# 일반적인 contrastive learning만 사용 (optional)
```

**차이점**:
- ❌ 상용 VLM: VICReg **미사용**
- ✅ PanoLLaVA: **VICReg 기반** 자기 감시 학습

---

### 3. **Resampler 구조 선택** ⭐⭐

**PanoLLaVA (default.yaml)**:
```yaml
resampler_type: "bimamba"  # BiDirectional Mamba
```

**지원 옵션**:
- `mlp`: 간단한 MLP (빠름)
- `qformer`: BLIP2 스타일 (정확함)
- `bimamba`: 양방향 Mamba (최신, 빠름)
- `perceiver`: Perceiver IO (유연함)

**상용 VLM**:
```yaml
# 대부분 고정
resampler: "linear_projection"  # 선형 프로젝션만
```

**차이점**:
- ❌ 상용 VLM: **선형 프로젝션** 고정
- ✅ PanoLLaVA: **5가지 구조** 중 선택 가능

---

### 4. **Positional Encoding (Projection Layer)** ⭐

**PanoLLaVA (default.yaml)**:
```yaml
use_projection_positional_encoding: true
# pe_view_encoding_type: "sinusoidal"      # 뷰 위치 인코딩
# pe_spatial_encoding_type: "sinusoidal"   # 공간 위치 인코딩
# pe_enable_continuity: true               # 360° 연속성
```

**의미**:
- 다중 뷰의 **상대 위치** 정보 인코딩
- 360° 파노라마의 **순환 구조** 반영
- Projection layer에서 위치 정보 추가

**상용 VLM**:
```yaml
# 기본 Vision Transformer의 positional encoding만 사용
# 다중 뷰 위치 정보는 고려 안 함
```

**차이점**:
- ❌ 상용 VLM: **단일 뷰** PE만
- ✅ PanoLLaVA: **다중 뷰 위치** + **360° 연속성** PE

---

### 5. **Vision Encoder Fine-tuning 제어** ⭐

**PanoLLaVA (default.yaml)**:
```yaml
vision:
  vision_trainable_blocks: 2  # 마지막 2개 블록 학습

resampler:
  vision_trainable_blocks: 0  # 완전 동결

finetune:
  vision_trainable_blocks: 0  # 완전 동결
```

**의미**:
- **Stage별로 다르게 설정** 가능
- Vision stage: 마지막 2개 블록만 학습 (VICReg 최적화)
- Resampler/Finetune: 완전 동결 (안정성)

**상용 VLM**:
```yaml
# 일반적으로
vision_encoder: frozen  # 항상 동결
# 또는
vision_encoder: trainable  # 전체 학습
```

**차이점**:
- ❌ 상용 VLM: **전부 동결** 또는 **전부 학습**
- ✅ PanoLLaVA: **부분 학습** + **Stage별 제어**

---

### 6. **Batch Size & Accumulation 전략** ⚠️

**PanoLLaVA (default.yaml)**:
```yaml
vision:
  batch_size: 16               # 큰 배치
  accumulate_grad_batches: 2   # 실효 배치: 32

resampler:
  batch_size: 1                # 작은 배치
  accumulate_grad_batches: 2   # 실효 배치: 2

finetune:
  batch_size: 1                # 작은 배치
  accumulate_grad_batches: 2   # 실효 배치: 2
```

**상용 VLM**:
```yaml
# 일반적으로
batch_size: 64-256  # 큰 배치 (데이터센터 환경)
accumulate_grad_batches: 1  # 직접 업데이트
```

**차이점**:
- ❌ 상용 VLM: **큰 배치** + **강력한 하드웨어**
- ⚠️ PanoLLaVA: **작은 배치** (resampler/finetune) → **메모리 제약**

**문제점**: resampler와 finetune의 batch_size=1은 **너무 작음** → 불안정

---

### 7. **데이터셋 구성** ⭐

**PanoLLaVA (default.yaml)**:
```yaml
vision:
  data:
    csv_train:
      - "data/quic360/train.csv"           # 파노라마
      - "data/train_stanford_dummy_anno.csv"  # 실내 파노라마
      - "data/train_zind_dummy_anno.csv"      # 실내 파노라마
    csv_val:
      - "data/quic360/valid.csv"
```

**의미**:
- **여러 파노라마 데이터셋** 혼합 학습
- Stage별로 다른 데이터 설정 가능

**상용 VLM**:
```yaml
data:
  train: ["large_mixed_dataset.json"]  # 수백만 장
  # 예: LLaVA-558K, ShareGPT4V, etc.
```

**차이점**:
- ❌ 상용 VLM: **일반 이미지** 대규모 데이터셋
- ✅ PanoLLaVA: **파노라마 전용** 소규모 데이터셋

---

## 📊 종합 비교표

| 설정 항목 | PanoLLaVA (default.yaml) | 상용 VLM | 공통점/차이점 |
|----------|-------------------------|----------|-------------|
| **Vision Encoder** | SigLIP2-SO400M | SigLIP/RICE | ✅ 동일 계열 |
| **Language Model** | Qwen3-0.6B | Qwen2/Llama | ✅ 동일 계열 |
| **Resampler** | BiMamba (5가지 선택) | 선형 프로젝션 | ❌ 구조 다양성 |
| **이미지 처리** | anyres_e2p (파노라마) | resize/crop | ❌ 기하 변환 |
| **VICReg Loss** | ✅ (overlap) | ❌ | ❌ 파노라마 특화 |
| **Positional Encoding** | Multi-view PE | Single-view PE | ❌ 다중 뷰 위치 |
| **Vision 학습** | 부분 학습 (블록 2개) | 전부 동결/학습 | ❌ 세밀한 제어 |
| **LoRA** | ✅ (rank=32) | 대부분 ❌ | ✅ PEFT 트렌드 |
| **Multi-Stage** | 3단계 명시적 | 2-3단계 | ✅ Progressive |
| **Batch Size** | 16 → 1 → 1 | 64-256 | ⚠️ 메모리 제약 |
| **데이터** | 파노라마 전용 | 일반 이미지 | ❌ 도메인 특화 |

---

## 🎯 핵심 결론

### **공통점 (표준 VLM 설계)**

1. ✅ **Vision Encoder**: SigLIP/CLIP 계열 사용
2. ✅ **Language Model**: Transformer 기반 LLM (Qwen/Llama)
3. ✅ **Multi-Stage Training**: 점진적 학습 전략
4. ✅ **Flash Attention**: 메모리 최적화
5. ✅ **Small Model Support**: 0.5B-7B 범위

### **차이점 (PanoLLaVA 독창성)** ⭐

1. ❌ **파노라마 특화 처리**: anyres_e2p (E2P 타일화)
2. ❌ **VICReg Overlap Loss**: 인접 뷰 정규화
3. ❌ **Resampler 다양성**: 5가지 구조 선택
4. ❌ **Multi-view Positional Encoding**: 360° 연속성
5. ❌ **세밀한 Vision Encoder 제어**: 블록별 학습/동결
6. ⚠️ **작은 배치 크기**: 메모리 제약 (개선 필요)

---

## 💡 개선 제안

### 1. **Batch Size 증가** (중요)

**현재 문제**:
```yaml
resampler:
  batch_size: 1  # ❌ 너무 작음
  
finetune:
  batch_size: 1  # ❌ 너무 작음
```

**권장 수정**:
```yaml
resampler:
  batch_size: 4-8  # ✅ 안정적 학습
  accumulate_grad_batches: 4  # 실효 배치: 16-32
  
finetune:
  batch_size: 4-8  # ✅ 안정적 학습
  accumulate_grad_batches: 4  # 실효 배치: 16-32
```

### 2. **데이터 경로 통일**

**현재 문제**: 3곳에 중복 정의
```yaml
paths:
  csv_train: "data/quic360/train.csv"  # 1번
  
vision:
  data:
    csv_train: [...]  # 2번
    
data:
  train: [...]  # 3번
```

**권장**: Stage별 설정을 우선하고, 나머지는 제거

### 3. **Vision Encoder Fine-tuning 전략**

**현재 설정**: vision stage만 2개 블록 학습
```yaml
vision:
  vision_trainable_blocks: 2  # ✅ OK
  
finetune:
  vision_trainable_blocks: 0  # ⚠️ 고려: 2-4로 증가?
```

**대안**: Finetune stage에서도 일부 블록 학습 (end-to-end)

---

## 📚 참고: 상용 VLM 설정 예시

### LLaVA-OneVision

```yaml
vision_encoder: rice-vit-large (300M)
language_model: Qwen2-4B
projection: linear (1024 → 4000)
training_stages: [pretraining, SFT]
batch_size: 256
data: LLaVA-558K + custom
```

### Qwen2.5-VL

```yaml
vision_encoder: SigLIP-Large (427M)
language_model: Qwen2.5-7B
projection: linear (4096 → 5120)
training_stages: [vision_pretrain, SFT]
batch_size: 128
data: 대규모 일반 이미지
```

---

## 결론

**PanoLLaVA의 default.yaml**은:
- ✅ **표준 VLM 설계 원칙**을 따르면서도
- ⭐ **파노라마 특화 기능**을 추가한 **하이브리드 설계**
- ⚠️ 일부 설정 (batch_size) 개선 필요
- 🎯 **360° 이미지 이해**에 최적화된 독창적 아키텍처
