# 🎨 이미지 프로세서 시각화 가이드

## 개요

`visualize_processors.py`는 PanoLLaVA의 여러 이미지 처리 방식(crop strategy)이 파노라마 이미지를 어떻게 변환하는지 시각화하는 도구입니다.

## 지원하는 처리 방식

### 1. **RESIZE** - 단순 리사이징
- **특징**: 가장 간단한 방식, 파노라마를 단일 이미지로 리사이징
- **뷰 수**: 1개
- **용도**: 기본 베이스라인, 非파노라마 이미지 처리
- **장점**: 빠름, 메모리 효율적
- **단점**: 파노라마의 360도 정보 손실

```
원본 파노라마 (2048×1024)
        ↓
     Resize
        ↓
   단일 뷰 (224×224)
```

### 2. **CUBEMAP** - 큐브맵 투영
- **특징**: 파노라마를 6개 면(정면, 우측, 좌측, 상단, 하단, 후면) 중 4개로 분할
- **뷰 수**: 4개 (정면, 우측, 좌측, 후면)
- **용도**: 공간적 일관성이 중요한 경우
- **장점**: 각 뷰의 높은 품질, 각도별 특성 보존
- **단점**: 상하 극점(pole) 정보 손실, 경계 이음선 가능성

```
원본 파노라마 (Equirectangular)
        ↓
  큐브맵 투영 (각 각도별)
        ↓
  4개 뷰 [Front, Right, Left, Back]
```

### 3. **SLIDING WINDOW** - 슬라이딩 윈도우
- **특징**: 파노라마를 가로로 슬라이딩하여 여러 중첩되는 뷰 추출
- **뷰 수**: 보통 6-8개 (이미지 크기와 overlap ratio에 따라)
- **용도**: 연속적인 파노라마 커버리지, 부드러운 전환
- **장점**: 360도 연속 커버, 부드러운 오버랩
- **단점**: 중복 처리로 인한 계산 비용

```
원본 파노라마 (Equirectangular)
        ↓
  Sliding Window (FOV: 90°, Overlap: 50%)
        ↓
  8개 뷰 [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]
```

### 4. **ANYRES-E2P** - AnyRes 스타일 ERP 타일링 (권장)
- **특징**: 전역(글로벌) 뷰 + 여러 타일로 파노라마 커버
- **뷰 수**: 1개 글로벌 + 여러 개 타일 (보통 16-20개)
- **용도**: 고해상도 처리, 전체 파노라마 커버리지 필요할 때
- **장점**: 
  - 360도 전체 커버 (폐곡선 순환)
  - 전역 + 상세 정보 병렬 처리
  - 겹침(overlap) 영역으로 부드러운 전환
- **단점**: 계산 비용 높음, 더 많은 메모리 필요

```
원본 파노라마 (Equirectangular)
        ↓
  ERP → Pinhole 타일 변환
        ↓
  1개 전역 뷰 (336×336)
  + 타일 뷰들 (보통 4×4 격자)
```

## 사용법

### 기본 사용 (이미지 저장 + 시각화)

```bash
conda activate pano
python scripts/visualize_processors.py --image-path <이미지경로>
```

**예시:**
```bash
python scripts/visualize_processors.py --image-path data/quic360/downtest/images/2094501355_045ede6d89_k.jpg
```

### 출력 경로 지정

```bash
python scripts/visualize_processors.py \
  --image-path <이미지경로> \
  --output <출력경로>
```

**예시:**
```bash
python scripts/visualize_processors.py \
  --image-path data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
  --output results/processor_views/
```

### 이미지 크기 지정

```bash
python scripts/visualize_processors.py \
  --image-path <이미지경로> \
  --size <크기>
```

**예시:**
```bash
# 336×336 크기로 처리
python scripts/visualize_processors.py \
  --image-path data/sample.jpg \
  --size 336
```

### 시각화만 생성 (개별 이미지 저장 제외)

```bash
python scripts/visualize_processors.py \
  --image-path <이미지경로> \
  --viz-only
```

## 출력 형식

생성되는 결과는 다음과 같습니다:

### 1. 비교 시각화 (compare_visualization.png)
모든 처리 방식의 뷰들을 하나의 이미지에 비교 표시

```
┌─────────────────────────────────────────┐
│     원본 파노라마 이미지                 │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ RESIZE (1개 뷰)                         │
│  [단일 이미지]                          │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ CUBEMAP (4개 뷰)                        │
│  [Front] [Right] [Left] [Back]          │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ SLIDING_WINDOW (8개 뷰)                 │
│  [View0] [View1] ... [View7]            │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ ANYRES-E2P (17개 뷰: 글로벌+타일)       │
│  [Global]                               │
│  [Tile 0] [Tile 1] ... [Tile 15]       │
└─────────────────────────────────────────┘
```

### 2. 각 방식별 폴더 구조

```
output_dir/
├── resize/
│   └── resize_view_000.png          (1개 이미지)
├── cubemap/
│   ├── cubemap_view_000.png         (4개 이미지)
│   ├── cubemap_view_001.png
│   ├── cubemap_view_002.png
│   └── cubemap_view_003.png
├── sliding_window/
│   ├── sliding_window_view_000.png  (8개 이미지)
│   ├── sliding_window_view_001.png
│   ├── ...
│   └── sliding_window_view_007.png
└── anyres_e2p/
    ├── anyres_e2p_view_000.png      (17개 이미지)
    ├── anyres_e2p_view_001.png
    ├── ...
    └── anyres_e2p_view_016.png
```

## 실제 예시

### 전형적인 실행 결과

```
📷 원본 이미지 로드: data/quic360/downtest/images/2094501355_045ede6d89_k.jpg
   이미지 크기: (2048, 1024)
======================================================================
📊 이미지 프로세서 시각화 시작
======================================================================

🔄 처리 중: RESIZE
  ✓ 성공: 1개 뷰 생성
    이미지 형태: torch.Size([1, 3, 224, 224])

🔄 처리 중: CUBEMAP
  ✓ 성공: 4개 뷰 생성
    이미지 형태: torch.Size([4, 3, 224, 224])

🔄 처리 중: SLIDING_WINDOW
  ✓ 성공: 8개 뷰 생성
    이미지 형태: torch.Size([8, 3, 224, 224])

🔄 처리 중: ANYRES_E2P
  ✓ 성공: 17개 뷰 생성
    이미지 형태: torch.Size([9, 3, 336, 336])

💾 각 전략별 이미지 저장 중...
======================================================================

📁 RESIZE
   저장 경로: output_dir/resize
   ✓ 1개 이미지 저장 완료

📁 CUBEMAP
   저장 경로: output_dir/cubemap
   ✓ 4개 이미지 저장 완료

📁 SLIDING_WINDOW
   저장 경로: output_dir/sliding_window
   ✓ 8개 이미지 저장 완료

📁 ANYRES_E2P
   저장 경로: output_dir/anyres_e2p
   ✓ 9개 이미지 저장 완료

======================================================================
📈 처리 결과 요약
======================================================================

RESIZE:
  ✓ 상태: 성공
  - 뷰 수: 1

CUBEMAP:
  ✓ 상태: 성공
  - 뷰 수: 4

SLIDING_WINDOW:
  ✓ 상태: 성공
  - 뷰 수: 8

ANYRES_E2P:
  ✓ 상태: 성공
  - 뷰 수: 17

✅ 완료!
```

## 각 방식의 선택 가이드

| 방식 | 속도 | 메모리 | 정확도 | 뷰 수 | 사용 케이스 |
|------|------|--------|--------|-------|-----------|
| **RESIZE** | ⚡⚡⚡ | ⚡ | ⭐ | 1 | 빠른 추론, 非파노라마 이미지 |
| **CUBEMAP** | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ | 4 | 균형잡힌 처리 |
| **SLIDING_WINDOW** | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ | 6-8 | 연속 커버리지, 부드러운 전환 |
| **ANYRES-E2P** | ⚡ | ⚡⚡⚡ | ⭐⭐⭐⭐ | 17+ | 고정확도, VQA, 캡셔닝 |

## 통합 사용 예시

### 훈련 중 다양한 전략 비교

```python
from panovlm.processors.image import PanoramaImageProcessor
from PIL import Image

image = Image.open("path/to/panorama.jpg")

# 각 방식으로 처리
strategies = {
    'resize': (224, 224),
    'cubemap': (224, 224),
    'sliding_window': (224, 224),
    'anyres_e2p': (336, 336),
}

for strategy, size in strategies.items():
    processor = PanoramaImageProcessor(
        image_size=size,
        crop_strategy=strategy
    )
    views, metadata = processor(image, return_metadata=True)
    print(f"{strategy}: {views.shape}, {len(metadata)} views")
```

## 문제 해결

### 1. 이미지가 너무 큼

파노라마 이미지가 매우 큰 경우 `--size` 파라미터를 작게 지정하세요:

```bash
python scripts/visualize_processors.py \
  --image-path <이미지경로> \
  --size 224  # 기본값
```

### 2. 메모리 부족

AnyRes-E2P는 많은 메모리를 사용합니다. 필요시 타일 크기를 줄이거나 다른 전략을 사용하세요:

```bash
python scripts/visualize_processors.py \
  --image-path <이미지경로> \
  --size 224  # 작은 크기
```

### 3. 폰트 경고

한글 폰트 경고는 무시해도 괜찮습니다. 이미지는 정상적으로 저장됩니다.

## 추가 정보

- **입력 이미지 형식**: JPG, PNG (권장 1:2 종횡비, 예: 2048×1024)
- **최소 요구사항**: 512×256 이상
- **출력 형식**: PNG (고품질)
- **처리 시간**: 약 30-60초 (방식에 따라)
- **저장 용량**: 약 2-5MB (전체 방식 포함)
