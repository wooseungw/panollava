# Vision Encoder 시각화 가이드

이 가이드는 `dino.py`를 사용하여 vision encoder의 hidden states를 시각화하고 분석하는 방법을 설명합니다.

## 목차
1. [빠른 시작](#빠른-시작)
2. [상세 사용법](#상세-사용법)
3. [시각화 결과 해석](#시각화-결과-해석)
4. [Python API 사용](#python-api-사용)
5. [고급 옵션](#고급-옵션)

---

## 빠른 시작

### 1. 기본 사용법

```bash
# SigLIP 모델로 파노라마 이미지 시각화
python scripts/visualize_vision_encoder.py \
    --image data/quic360/train/pano_001.jpg \
    --vision_model google/siglip-base-patch16-224 \
    --crop_strategy e2p \
    --output_dir results/vision_viz/siglip_e2p
```

### 2. DINOv2 모델 사용

```bash
# DINOv2로 시각화
python scripts/visualize_vision_encoder.py \
    --image data/quic360/train/pano_001.jpg \
    --vision_model facebook/dinov2-base \
    --crop_strategy anyres_e2p \
    --output_dir results/vision_viz/dinov2_anyres
```

### 3. CLIP 모델 사용

```bash
# CLIP으로 시각화
python scripts/visualize_vision_encoder.py \
    --image data/quic360/train/pano_001.jpg \
    --vision_model openai/clip-vit-base-patch16 \
    --crop_strategy cubemap \
    --output_dir results/vision_viz/clip_cubemap
```

---

## 상세 사용법

### 명령줄 인자

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--image` | ✅ | - | 입력 이미지 경로 |
| `--vision_model` | ❌ | `google/siglip-base-patch16-224` | Vision encoder 모델 |
| `--crop_strategy` | ❌ | `e2p` | 이미지 crop 전략 |
| `--image_size` | ❌ | `224` | 입력 이미지 크기 |
| `--output_dir` | ❌ | `results/vision_viz` | 출력 디렉토리 |
| `--n_components` | ❌ | `3` | PCA 주성분 개수 |
| `--no_cls_token` | ❌ | - | CLS 토큰 제거 안 함 |
| `--bg_removal` | ❌ | `threshold` | 배경 제거 방법 |
| `--no_similarity` | ❌ | - | 유사도 분석 건너뛰기 |
| `--device` | ❌ | `auto` | 디바이스 (auto/cuda/cpu) |

### Crop 전략

- **`e2p`**: Equirectangular-to-Perspective (중앙 90° FOV)
- **`anyres_e2p`**: AnyRes 스타일 ERP 타일링 (yaw/pitch 그리드)
- **`cubemap`**: 4면 큐브맵 (앞/뒤/좌/우)
- **`sliding_window`**: 수평 슬라이딩 윈도우
- **`anyres`**: 그리드 기반 패치
- **`resize`**: 단순 리사이즈 (베이스라인)

### 배경 제거 방법

- **`threshold`**: 첫 번째 PCA 성분에 임계값 적용 (기본값)
- **`remove_first_pc`**: 첫 번째 주성분 완전 제거
- **`outlier_removal`**: Mahalanobis 거리 기반 이상치 제거
- **`none`**: 배경 제거 안 함

---

## 시각화 결과 해석

### 출력 파일

실행 후 다음 파일들이 생성됩니다:

```
results/vision_viz/
├── pca_visualization.png       # PCA RGB 시각화
├── original_views.png          # 원본 view 이미지
└── similarity_analysis.txt     # 유사도 분석 결과
```

### PCA 시각화 이해하기

**`pca_visualization.png`**:
- 각 view의 hidden states를 3개 주성분으로 압축하여 RGB로 표현
- **빨강 채널**: 첫 번째 주성분 (가장 큰 분산)
- **초록 채널**: 두 번째 주성분
- **파랑 채널**: 세 번째 주성분

**색상 패턴 해석**:
- 비슷한 색상 = 비슷한 semantic features
- 색상 대비 = feature space에서의 차이
- 배경(단조로운 영역) vs 전경(복잡한 영역) 구분 가능

### 유사도 분석 지표

**토큰 레벨 유사도** (`similarity_analysis.txt`):
- **MSE (Mean Squared Error)**: 낮을수록 유사 (0에 가까울수록 좋음)
- **Cosine**: 높을수록 유사 (1에 가까울수록 좋음)
- **Hungarian**: 최적 매칭 후 코사인 유사도
- **CKA (Centered Kernel Alignment)**: 표현 공간 정렬 측정

**PCA-RGB 이미지 유사도**:
- **MSE**: 픽셀 레벨 차이
- **SSIM**: 구조적 유사도 (0~1, 높을수록 유사)
- **LPIPS**: 지각적 유사도 (낮을수록 유사)

---

## Python API 사용

### 기본 사용 예시

```python
from panovlm.evaluation.dino import DinoVisualizer
from panovlm.processors.image import PanoramaImageProcessor
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch

# 1. 이미지 로딩 및 전처리
image = Image.open("data/quic360/train/pano_001.jpg")

vision_model = AutoModel.from_pretrained(
    "google/siglip-base-patch16-224"
).cuda()
vision_model.eval()

hf_processor = AutoImageProcessor.from_pretrained(
    "google/siglip-base-patch16-224"
)

pano_processor = PanoramaImageProcessor(
    crop_strategy="e2p",
    image_size=224,
    use_vision_processor=True,
    vision_processor=hf_processor
)

pixel_values = pano_processor(image).unsqueeze(0)  # [V, C, H, W]

# 2. Hidden states 추출
hidden_states_list = []
with torch.no_grad():
    for i in range(pixel_values.shape[0]):
        view = pixel_values[i:i+1].cuda()
        outputs = vision_model(view, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        hidden_states_list.append(last_hidden.cpu().numpy())

# 3. DinoVisualizer로 시각화
visualizer = DinoVisualizer(
    hidden_states_list=hidden_states_list,
    remove_cls_token=True
)

# 4. PCA 학습
visualizer.fit_pca(
    n_components=3,
    use_background_removal=True,
    bg_removal_method="threshold"
)

# 5. 결과 플롯
visualizer.plot_pca_results(
    titles=[f'View {i+1}' for i in range(len(hidden_states_list))],
    save_path="results/my_pca_viz.png"
)

# 6. 유사도 분석
pairs = [(0, 1), (1, 2), (2, 3)]
token_sim = visualizer.get_token_similarity(pairs=pairs)
pca_sim = visualizer.get_pca_similarity(pairs=pairs)

print("토큰 레벨 코사인 유사도:", token_sim['cosine'])
print("PCA SSIM:", pca_sim['ssim'])
```

### 고급: 커스텀 분석

```python
# 특정 레이어의 hidden states 추출
layer_idx = 6  # 중간 레이어

with torch.no_grad():
    for view in pixel_values:
        outputs = vision_model(view.unsqueeze(0).cuda(), output_hidden_states=True)
        layer_hidden = outputs.hidden_states[layer_idx]
        hidden_states_list.append(layer_hidden.cpu().numpy())

# 더 많은 주성분 사용
visualizer.fit_pca(n_components=10)

# PCA 설명 분산 확인
explained_var = visualizer.pca_model.explained_variance_ratio_
print(f"상위 10개 주성분 설명 분산: {explained_var.sum():.2%}")
```

---

## 고급 옵션

### 1. 여러 모델 비교

```bash
# 스크립트 작성: compare_models.sh
for model in \
    "google/siglip-base-patch16-224" \
    "facebook/dinov2-base" \
    "openai/clip-vit-base-patch16"
do
    model_name=$(echo $model | tr '/' '_')
    python scripts/visualize_vision_encoder.py \
        --image data/quic360/train/pano_001.jpg \
        --vision_model $model \
        --crop_strategy e2p \
        --output_dir results/vision_viz/$model_name
done
```

### 2. 배치 처리

```python
from pathlib import Path
import subprocess

image_dir = Path("data/quic360/train")
output_base = Path("results/vision_viz_batch")

for img_path in image_dir.glob("*.jpg")[:10]:  # 처음 10개만
    output_dir = output_base / img_path.stem
    
    subprocess.run([
        "python", "scripts/visualize_vision_encoder.py",
        "--image", str(img_path),
        "--vision_model", "google/siglip-base-patch16-224",
        "--output_dir", str(output_dir)
    ])
```

### 3. 다양한 crop 전략 비교

```bash
for strategy in e2p anyres_e2p cubemap sliding_window
do
    python scripts/visualize_vision_encoder.py \
        --image data/quic360/train/pano_001.jpg \
        --crop_strategy $strategy \
        --output_dir results/vision_viz/strategy_$strategy
done
```

### 4. Warp 기반 유사도 분석 (고급)

```python
from panovlm.evaluation.dino import compute_overlap_consistency_score

# ERP 좌표 생성 및 warp 기반 유사도 계산
# (dino.py의 compute_overlap_consistency_score 함수 참조)

hidden_tensor = torch.from_numpy(hidden_states_list[0])  # [1, seq_len, dim]
ocs_result = compute_overlap_consistency_score(
    hidden_tensor,
    yaw_offset_deg=45.0,  # 45도 회전
    overlap_ratio=0.5
)

print(f"Overlap Consistency Score: {ocs_result['ocs']:.4f}")
print(f"Residual Mean: {ocs_result['residual_mean']:.4f}")
```

---

## 트러블슈팅

### LPIPS 사용 불가

```bash
# LPIPS 설치
pip install lpips

# 또는 LPIPS 없이 실행 (자동으로 건너뜀)
python scripts/visualize_vision_encoder.py --image ... --no_similarity
```

### CUDA Out of Memory

```bash
# CPU 사용
python scripts/visualize_vision_encoder.py --image ... --device cpu

# 또는 이미지 크기 줄이기
python scripts/visualize_vision_encoder.py --image ... --image_size 128
```

### 한글 폰트 깨짐

```python
# dino.py의 _setup_korean_font() 함수가 자동으로 처리
# 필요시 시스템에 한글 폰트 설치:
# Ubuntu: sudo apt-get install fonts-nanum
# macOS: 시스템 기본 폰트 사용
```

---

## 참고 자료

- **DINOv2 논문**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **PCA 시각화**: Vision Transformer의 주성분이 semantic features를 잘 포착함
- **CKA 유사도**: [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)

---

## 예시 결과

### SigLIP (E2P strategy)
```
📊 PCA 분석 결과
주성분 1 설명 분산: 18.45%
주성분 2 설명 분산: 12.32%
주성분 3 설명 분산: 8.76%
총 설명 분산 (상위 3개): 39.53%

🔍 토큰 레벨 유사도:
  Pair (0, 1): Cosine=0.8234
  평균: 0.8234

🎨 PCA-RGB 이미지 유사도:
  Pair (0, 1): SSIM=0.7456
  평균: 0.7456
```

### DINOv2 (AnyRes ERP strategy)
```
📊 PCA 분석 결과
주성분 1 설명 분산: 22.14%
주성분 2 설명 분산: 15.67%
주성분 3 설명 분산: 11.23%
총 설명 분산 (상위 3개): 49.04%

🔍 토큰 레벨 유사도:
  여러 view 간 평균 코사인 유사도: 0.7892
```

---

**Happy Visualizing! 🎨🔍**
