# AnyRes ERP Integration Guide

## 개요

이 문서는 `panovlm.processors.anyres_e2p` 모듈을 PanoVLM의 이미지 처리 및 VICReg loss 계산에 통합하는 방법을 설명합니다.

## 주요 변경사항

### 1. PanoramaImageProcessor에 anyres_e2p 전략 추가

**위치**: `src/panovlm/data/image.py`

#### 새로운 crop_strategy: `anyres_e2p`

```python
from panovlm.data.image import PanoramaImageProcessor

processor = PanoramaImageProcessor(
    image_size=(224, 224),
    crop_strategy="anyres_e2p",  # 새로운 전략
    fov_deg=90.0,
    overlap_ratio=0.5,

    # AnyRes ERP 전용 파라미터
    anyres_e2p_base_size=336,           # 전역 이미지 크기
    anyres_e2p_tile_size=672,           # 타일 렌더 크기
    anyres_e2p_vit_size=224,            # 최종 ViT 입력 크기
    anyres_e2p_closed_loop=True,        # 360도 폐곡선 분할
    anyres_e2p_pitch_range=(-45.0, 45.0)  # pitch 범위
)

# 이미지 처리
views_tensor = processor(image_path)  # [V, 3, H, W]
# or
views_tensor, metadata = processor(image_path, return_metadata=True)

# 타일 메타데이터 접근
tile_metas = processor.tile_metas
for meta in tile_metas:
    print(f"Tile {meta['tile_id']}: yaw={meta['yaw_deg']:.1f}°, pitch={meta['pitch_deg']:.1f}°")
```

### 2. VICReg Loss의 Grid-based Pairing

**위치**: `src/panovlm/data/anyres_integration.py`

#### 기존 문제점
- 기존 VICReg는 **순차적 인접 뷰(v, v+1)**만 고려
- AnyRes ERP는 **2D grid 구조**의 타일 배치

#### 해결 방법: AnyResVICRegPairing

```python
from panovlm.data.anyres_integration import AnyResVICRegPairing

pairing = AnyResVICRegPairing(
    hfov_deg=90.0,
    vfov_deg=90.0,
    overlap_ratio=0.5,
    pairing_strategy="adjacent",  # 'adjacent' | 'distance_based' | 'all_pairs'
    distance_threshold=120.0
)

# 타일 메타데이터로부터 페어 생성
pairs = pairing.build_tile_pairs(tile_metas)

for idx1, idx2, overlap_ratio in pairs:
    print(f"Pair: ({idx1}, {idx2}) - overlap: {overlap_ratio:.2f}")
```

#### Pairing Strategies

**1. `adjacent` (권장)**
- 2D grid에서 공간적으로 인접한 타일만 페어링
- 수평 인접: 같은 pitch, 인접한 yaw
- 수직 인접: 같은 yaw, 인접한 pitch
- 360도 순환 연결 지원 (파노라마 특성)

**2. `distance_based`**
- 구면 거리(Great Circle Distance) 기반
- `distance_threshold` 내의 모든 타일 페어

**3. `all_pairs`**
- 모든 가능한 페어 (메모리 집약적)

### 3. Model 통합

**위치**: `src/panovlm/models/model.py`

#### Config 설정

```python
from panovlm.config.config_manager import ModelConfig

config = ModelConfig(
    vision_name="google/siglip-base-patch16-224",
    language_model_name="Qwen/Qwen2.5-0.5B-Instruct",

    # VICReg 기본 설정
    vicreg_loss_weight=1.0,
    overlap_ratio=0.5,
    vicreg_mode="pairwise",

    # AnyRes ERP VICReg 활성화
    use_anyres_e2p_vicreg=True,
    anyres_vicreg_pairing_strategy="adjacent",  # 'adjacent' | 'distance_based' | 'all_pairs'
)

model = PanoramaVLM(config=config)
```

#### VICReg Loss 자동 전환

모델은 `tile_metas_batch`가 설정되어 있으면 자동으로 AnyRes ERP VICReg 모드로 전환됩니다:

```python
# Vision stage forward
loss_dict = model(
    pixel_values=images,  # [B, V, 3, H, W]
    stage="vision"
)

# model._compute_vicreg_overlap_loss() 내부에서:
# - tile_metas_batch가 있으면 → compute_vicreg_anyres_loss()
# - 없으면 → compute_vicreg_overlap_loss() (기존 sequential 방식)
```

## 완전한 사용 예시

### Dataset에서 tile_metas 전달

```python
from torch.utils.data import Dataset
import torch

class PanoramaDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # AnyRes ERP 처리
        views_tensor, metadata = self.processor(image_path, return_metadata=True)

        # 타일 메타데이터 반환
        tile_metas = self.processor.tile_metas if hasattr(self.processor, 'tile_metas') else []

        return {
            'pixel_values': views_tensor,
            'tile_metas': tile_metas
        }

# Collate function
def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    tile_metas_batch = [item['tile_metas'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'tile_metas_batch': tile_metas_batch
    }

# DataLoader
dataset = PanoramaDataset(image_paths, processor)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

### Training Loop

```python
for batch in dataloader:
    pixel_values = batch['pixel_values'].to(device)  # [B, V, 3, H, W]
    tile_metas_batch = batch['tile_metas_batch']

    # 모델에 tile_metas 전달
    model.tile_metas_batch = tile_metas_batch

    # Forward (VICReg loss 자동 계산)
    outputs = model(
        pixel_values=pixel_values,
        stage="vision"
    )

    loss = outputs['loss']
    loss.backward()
    optimizer.step()

    # 배치 끝에 초기화
    model.tile_metas_batch = []
```

## Configuration 옵션 요약

### PanoramaImageProcessor

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `crop_strategy` | `"e2p"` | `"anyres_e2p"` 설정 |
| `fov_deg` | `90.0` | 타일 FOV |
| `overlap_ratio` | `0.5` | 타일 겹침 비율 |
| `anyres_e2p_base_size` | `336` | 전역 이미지 크기 |
| `anyres_e2p_tile_size` | `672` | 타일 렌더 크기 |
| `anyres_e2p_vit_size` | `None` | 최종 ViT 크기 |
| `anyres_e2p_closed_loop` | `True` | 360도 폐곡선 분할 |
| `anyres_e2p_pitch_range` | `(-45.0, 45.0)` | pitch 범위 |

### ModelConfig

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `use_anyres_e2p_vicreg` | `False` | AnyRes ERP VICReg 활성화 |
| `anyres_vicreg_pairing_strategy` | `"adjacent"` | 페어링 전략 |
| `vicreg_loss_weight` | `1.0` | VICReg loss 가중치 |
| `overlap_ratio` | `0.5` | 겹침 비율 |

## 기존 코드와의 호환성

- 기존 `crop_strategy` (`"e2p"`, `"sliding_window"`, etc.)는 **그대로 동작**
- `use_anyres_e2p_vicreg=False`이면 **기존 sequential VICReg 사용**
- `tile_metas_batch`가 비어있으면 **자동으로 fallback**

## 디버깅

### VICReg 페어 확인

```python
from panovlm.data.anyres_integration import AnyResVICRegPairing

pairing = AnyResVICRegPairing(
    hfov_deg=90.0,
    overlap_ratio=0.5,
    pairing_strategy="adjacent"
)

pairs = pairing.build_tile_pairs(tile_metas)
print(f"Total pairs: {len(pairs)}")

for idx1, idx2, overlap in pairs[:5]:  # 처음 5개만
    meta1 = tile_metas[idx1]
    meta2 = tile_metas[idx2]
    print(f"  Pair ({idx1}, {idx2}): "
          f"yaw=({meta1['yaw_deg']:.1f}°, {meta2['yaw_deg']:.1f}°), "
          f"pitch=({meta1['pitch_deg']:.1f}°, {meta2['pitch_deg']:.1f}°), "
          f"overlap={overlap:.2f}")
```

### VICReg Loss 모드 확인

```python
# Training 중
print(f"Using AnyRes ERP VICReg: {model.use_anyres_e2p_vicreg}")
print(f"Tile metas available: {len(model.tile_metas_batch) > 0}")
print(f"Pairing strategy: {model.anyres_vicreg_pairing_strategy}")
```

## 성능 최적화

### 메모리 사용량 제어

```python
# pair_chunk 조정 (공분산 계산 시 메모리 피크 감소)
config = ModelConfig(
    # ... 기타 설정
    vicreg_pair_chunk=4,  # 기본값 8
)
```

### 페어링 전략 선택

- **적은 메모리, 빠른 학습**: `"adjacent"`
- **더 많은 supervision**: `"distance_based"` (threshold 조정)
- **최대 supervision** (비권장): `"all_pairs"`

## 참고 자료

- `src/panovlm/processors/anyres_e2p.py`: AnyRes ERP 타일 생성
- `src/panovlm/data/anyres_integration.py`: VICReg 페어링 로직
- `src/panovlm/data/image.py`: 이미지 처리 통합
- `src/panovlm/models/model.py`: 모델 통합

## Troubleshooting

### ImportError: anyres_e2p module not available

```bash
# scripts 디렉토리가 PYTHONPATH에 있는지 확인
export PYTHONPATH=$PYTHONPATH:/path/to/panollava/scripts

# 또는 py360convert 설치 확인
pip install py360convert opencv-python
```

### VICReg loss가 0이 됨

- `tile_metas_batch`가 제대로 전달되었는지 확인
- Global view (tile_id=0)는 자동으로 제외됨
- 최소 2개 이상의 타일 필요

### 메모리 부족

- `pair_chunk` 값 줄이기 (8 → 4 → 2)
- `pairing_strategy="adjacent"` 사용 (페어 수 감소)
- 배치 크기 줄이기

## TODO

- [ ] Config에서 `hfov_deg` 가져오기 (현재 하드코딩: 90.0)
- [ ] Lightning DataModule과의 통합 예시
- [ ] AnyRes ERP visualization 스크립트
- [ ] 성능 벤치마크 (sequential vs grid-based VICReg)

## 최근 업데이트 (2025-01-08)

### ✅ compute_vicreg_anyres_loss 수정 완료

**문제점:**
- 기존: Feature를 1D로 flatten하여 공간 구조 손실
- 기존: 단일 페어에 대해 variance/covariance 계산 (부정확)
- 기존: 중앙 토큰만 선택, 실제 geometric overlap과 무관

**수정 내용:**
- ✅ 2D grid 구조 유지: `[B, V, H, W, D]` → overlap 영역 추출
- ✅ Batch 단위 처리: 모든 페어를 `[P, L, D]`로 stack
- ✅ Pairwise variance/covariance: `compute_vicreg_overlap_loss`와 동일한 방식
- ✅ 메모리 효율적: covariance 청킹 지원

**결과:**
기존 sequential VICReg과 동일한 품질의 loss 계산!

### 수정 전후 비교

| 항목 | 수정 전 ❌ | 수정 후 ✅ |
|-----|----------|----------|
| Feature 구조 | 1D flatten `[1, L*D]` | 2D preserved `[P, L, D]` |
| Variance 계산 | 단일 페어 | Pairwise (batch) |
| Covariance 계산 | 단일 페어 | Pairwise (batch) |
| 메모리 효율 | N/A | 청킹 지원 |
| 기존 코드와 일관성 | 낮음 | 높음 |

