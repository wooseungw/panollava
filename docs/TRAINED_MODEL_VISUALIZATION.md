# 학습된 모델 Vision Encoder 시각화 가이드

학습된 PanoLLaVA 체크포인트에서 Vision Encoder를 추출하고 DINOv2 스타일로 시각화

## 📋 개요

`visualize_trained_model.py`는 학습된 체크포인트에서 Vision Encoder와 Resampler를 추출하여 feature space를 시각화합니다.

### 주요 기능

✨ **Vision Encoder Hidden States 시각화**: 학습된 vision backbone의 feature maps를 RGB로 변환  
✨ **Resampled Features 시각화**: Resampler를 거친 후의 features 시각화  
✨ **유사도 분석**: 인접 view 간 토큰 레벨, PCA-RGB 이미지 유사도 측정  
✨ **다양한 Crop Strategy 지원**: e2p, anyres_e2p, cubemap, sliding_window 등

## 🚀 빠른 시작

### 기본 사용

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg
```

### Config 파일과 함께

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_anyres_e2p_vision_mlp/last.ckpt \
    --config configs/default.yaml \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --crop_strategy anyres_e2p
```

## 📊 출력 결과

### 생성되는 파일

```
results/trained_viz/{checkpoint_name}/
├── vision_encoder_pca_{checkpoint_name}.png      # Vision Encoder PCA 시각화
├── resampled_features_pca_{checkpoint_name}.png  # Resampler 출력 시각화
├── original_views_{checkpoint_name}.png          # 입력 view 이미지
└── similarity_analysis_{checkpoint_name}.txt     # 유사도 분석 결과
```

### 1. Vision Encoder PCA 시각화

- **색상의 의미**: RGB 3개 주성분으로 변환된 feature space
  - 비슷한 색 = 비슷한 semantic features
  - 색상 대비가 큰 영역 = feature가 뚜렷이 구분되는 영역

### 2. Resampled Features 시각화

- Vision Encoder의 high-dimensional features를 latent dimension으로 압축한 결과
- Resampler(MLP/QFormer/Perceiver)가 어떤 정보를 보존하는지 확인 가능

### 3. 유사도 분석 메트릭

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| Token Cosine | 원본 feature 토큰 간 직접 비교 | [-1, 1] |
| Hungarian Cosine | 최적 매칭 후 유사도 (순서 불변) | [-1, 1] |
| Linear CKA | 표현 공간의 구조적 유사성 | [0, 1] |
| SSIM | PCA-RGB 이미지의 구조적 유사도 | [0, 1] |
| LPIPS | PCA-RGB 이미지의 지각적 거리 | [-∞, 0] |

## 🔧 주요 옵션

### Required Arguments

- `--checkpoint`: 학습된 체크포인트 경로 (.ckpt)
- `--image`: 입력 파노라마 이미지 경로

### Optional Arguments

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | None | 설정 파일 경로 (YAML) |
| `--crop_strategy` | None | Crop 전략 (None=config에서 자동) |
| `--image_size` | 224 | Vision encoder 입력 크기 |
| `--output_dir` | `results/trained_viz/{checkpoint_name}` | 출력 디렉토리 |
| `--n_components` | 3 | PCA 주성분 개수 (RGB용) |
| `--no_cls_token` | False | CLS 토큰 제거하지 않음 |
| `--bg_removal` | threshold | 배경 제거 방법 |
| `--no_similarity` | False | 유사도 분석 건너뛰기 |
| `--device` | auto | 디바이스 (auto/cuda/cpu) |

### 배경 제거 방법

- `threshold`: 낮은 분산을 가진 배경 영역 제거 (권장)
- `remove_first_pc`: 첫 번째 주성분(보통 배경)을 제거
- `outlier_removal`: 통계적 이상치 제거
- `none`: 배경 제거 없음

## 📝 사용 예시

### 1. E2P 전략 모델 시각화

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --crop_strategy e2p
```

**출력**: 1개 view (정면 90° FOV)

### 2. AnyRes ERP 전략 모델 시각화

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_anyres_e2p_vision_mlp/last.ckpt \
    --config configs/default.yaml \
    --image data/quic360/downtest/images/26279212771_33406eed0f_o.jpg
```

**출력**: 8+ views (yaw 방향 타일링)

### 3. QFormer Resampler 효과 확인

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_qformer/last.ckpt \
    --image data/quic360/downtest/images/2485001734_9a1a2d7e84_k.jpg \
    --crop_strategy e2p
```

**확인**: `resampled_features_pca_*.png`에서 QFormer의 attention 효과 확인

### 4. 여러 Stage 비교

```bash
# Vision stage
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --output_dir results/stage_comparison/vision

# Resampler stage
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_resampler_mlp/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --output_dir results/stage_comparison/resampler

# Finetune stage
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_finetune_qformer/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --output_dir results/stage_comparison/finetune
```

### 5. 고급 옵션 사용

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --n_components 3 \
    --bg_removal remove_first_pc \
    --no_cls_token \
    --device cuda \
    --output_dir results/custom_viz
```

## 🎨 시각화 해석

### Vision Encoder PCA 색상 패턴

#### 좋은 패턴 ✅

- **일관된 색상 그라데이션**: 비슷한 semantic 영역이 부드럽게 연결됨
- **명확한 경계**: 다른 객체 간 뚜렷한 색상 전환
- **View 간 유사성**: 겹치는 영역의 색상이 비슷함 (VICReg 효과)

#### 문제 패턴 ❌

- **노이즈가 많은 색상**: 학습이 불안정하거나 overfitting
- **단조로운 색상**: Feature diversity 부족, 정보 손실
- **View 간 불일치**: VICReg loss가 제대로 작동하지 않음

### Resampler 효과 분석

**MLP Resampler**:
- 단순한 linear projection
- Vision encoder features를 거의 그대로 유지
- 빠르지만 표현력 제한

**QFormer Resampler**:
- Cross-attention 기반 feature aggregation
- 더 semantic한 정보 추출
- 느리지만 표현력 높음

**비교 방법**:
```bash
# MLP
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../vision_mlp/last.ckpt --image test.jpg

# QFormer
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../vision_qformer/last.ckpt --image test.jpg
```

두 결과의 `resampled_features_pca_*.png`를 비교

## 📈 유사도 분석 활용

### similarity_analysis_*.txt 예시

```
==================================================
Vision Encoder - 토큰 레벨 유사도
==================================================

token_cosine:
  View 1 ↔ View 2: 0.8732
  View 2 ↔ View 3: 0.8654
  평균: 0.8693 ± 0.0039

hungarian_cosine:
  View 1 ↔ View 2: 0.9124
  평균: 0.9124 ± 0.0000

linear_cka:
  View 1 ↔ View 2: 0.7856
  평균: 0.7856 ± 0.0000
```

### 해석

- **Token Cosine > 0.85**: 매우 높은 유사도, VICReg 효과 좋음
- **Hungarian > Token**: 순서와 무관하게 유사한 features 존재
- **CKA > 0.7**: 표현 공간 구조가 유사함

## 🐛 문제 해결

### 1. Config 오류

```
ValueError: 체크포인트에서 config를 찾을 수 없습니다
```

**해결**: `--config` 옵션으로 설정 파일 지정

```bash
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../last.ckpt \
    --config configs/default.yaml \
    --image your_image.jpg
```

### 2. CUDA Out of Memory

```bash
# CPU 사용
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../last.ckpt \
    --image your_image.jpg \
    --device cpu

# 또는 이미지 크기 줄이기
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../last.ckpt \
    --image your_image.jpg \
    --image_size 192  # 224 대신
```

### 3. Import Error

```bash
# PYTHONPATH 설정
export PYTHONPATH=/data/1_personal/4_SWWOO/panollava/src:$PYTHONPATH

# 또는 패키지 설치
pip install -e .
```

### 4. LPIPS 없음

LPIPS는 선택 사항이며, 없어도 다른 메트릭은 계산됩니다:

```bash
# LPIPS 설치 (선택)
pip install lpips
```

## 🔬 연구/분석 활용

### VICReg Loss 효과 검증

```bash
# VICReg 적용 전 (초기 체크포인트)
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../epoch=0.ckpt \
    --image test.jpg \
    --output_dir results/vicreg_analysis/before

# VICReg 적용 후 (최종 체크포인트)
python scripts/visualize_trained_model.py \
    --checkpoint runs/.../last.ckpt \
    --image test.jpg \
    --output_dir results/vicreg_analysis/after
```

**비교**: `similarity_analysis_*.txt`의 유사도 증가 확인

### Crop Strategy 비교

```bash
for strategy in e2p anyres_e2p cubemap; do
    python scripts/visualize_trained_model.py \
        --checkpoint runs/checkpoint.ckpt \
        --image test.jpg \
        --crop_strategy $strategy \
        --output_dir results/crop_comparison/$strategy
done
```

### Batch 처리

```bash
# 여러 이미지 자동 처리
for img in data/quic360/downtest/images/*.jpg; do
    python scripts/visualize_trained_model.py \
        --checkpoint runs/.../last.ckpt \
        --image "$img" \
        --output_dir results/batch_viz/$(basename "$img" .jpg)
done
```

## 📚 참고 자료

- **DINO 논문**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **VICReg 논문**: [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
- **코드 참조**:
  - `src/panovlm/evaluation/dino.py`: DinoVisualizer 클래스
  - `src/panovlm/processors/image.py`: PanoramaImageProcessor
  - `scripts/visualize_trained_model.py`: 메인 스크립트

## 🎯 다음 단계

시각화 결과 확인 후:

1. ✅ **유사도 분석**: View 간 feature 일관성 평가
2. ✅ **Resampler 비교**: MLP vs QFormer 성능 비교
3. ✅ **다양한 이미지**: 일반화 성능 테스트
4. ✅ **VICReg 효과**: Overlap 영역 feature 유사도 검증
5. ✅ **Stage 비교**: Vision → Resampler → Finetune 변화 관찰
