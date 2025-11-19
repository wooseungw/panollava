# AnyRes E2P 코드 동작 원리

## 개요
`anyres_e2p.py`는 **Equirectangular Projection (ERP, 정방위도법)** 파노라마 이미지를 **Pinhole Camera (원근 투영)** 방식의 다중 타일로 변환하는 전처리 모듈입니다. LLaVA-NeXT의 AnyRes 전략을 360° 파노라마에 적용하여, 글로벌 컨텍스트와 고해상도 로컬 타일을 동시에 제공합니다.

---

## 핵심 구조

### 1. 유틸리티 함수들

#### `deg2rad(d: float) -> float`
- **목적**: 각도를 라디안으로 변환
- **사용처**: 삼각함수 계산 시 필수

#### `yaw_pitch_to_xyz(yaw_deg, pitch_deg) -> (x, y, z)`
- **목적**: 구면 좌표(yaw, pitch)를 3D 단위벡터로 변환
- **수식**:
  ```
  x = cos(pitch) * cos(yaw)
  y = sin(pitch)
  z = cos(pitch) * sin(yaw)
  ```
- **용도**: 각 타일의 중심 방향을 3D 벡터로 표현 (포지셔널 임베딩에 활용 가능)

#### `letterbox_square(img, size, fill) -> Image`
- **목적**: 이미지를 정사각형으로 만들면서 종횡비 유지
- **동작**:
  1. 긴 변을 `size`에 맞춤 (예: 800×600 → 336×252)
  2. 짧은 변은 검은색 패딩 추가 (336×252 → 336×336)
- **사용처**: 글로벌 이미지 생성 시 ViT 입력 요구사항 충족

#### `pil_to_tensor(img) -> torch.Tensor`
- **목적**: PIL Image를 PyTorch Tensor로 변환
- **과정**:
  ```python
  RGB PIL Image (H, W, 3, uint8 [0,255])
    ↓ np.array() / 255.0
  numpy (H, W, 3, float32 [0,1])
    ↓ permute(2,0,1)
  torch.Tensor (3, H, W, float32 [0,1])
  ```

#### `resize_to_vit(tensor, vit_size) -> torch.Tensor`
- **목적**: 타일을 ViT 입력 크기로 리사이즈
- **예시**: 672×672 타일 → 336×336 (bilinear interpolation)
- **조건**: `vit_size=None`이면 원본 크기 유지

---

## 2. 타일링 파라미터 계산

### FOV 계산
#### `compute_vfov_from_hfov(hfov_deg, out_size) -> float`
- **가정**: 정사각형 출력 (픽셀 종횡비 1:1)
- **결과**: `vFOV ≈ hFOV` (90° hFOV → 90° vFOV)

### 각도 정규화
#### `_norm_angle_180(x) -> float`
- **목적**: 각도를 `[-180, 180)` 범위로 정규화
- **예시**: 
  - 270° → -90°
  - -200° → 160°

---

## 3. 타일 중심 계산 전략

### 전략 A: 열린 구간 방식 (Standard)
#### `make_yaw_centers_standard(hfov_deg, overlap, yaw_start, yaw_end, phase_deg)`

**동작 원리**:
```
step = hfov * (1 - overlap)
예: hFOV=90°, overlap=0.5 → step=45°

시작점: yaw_start + hfov/2 + phase_deg
= -180° + 45° + 0° = -135°

중심 위치: -135°, -90°, -45°, 0°, 45°, 90°, 135°
(+180° 근처는 제외됨)
```

**문제점**: 360° 완전 커버리지가 불가능 (경계 누락)

### 전략 B: 폐곡선 방식 (Closed Loop) ⭐ **권장**
#### `make_yaw_centers_closed_loop(hfov_deg, overlap, start_deg, seam_phase_deg)`

**동작 원리**:
```
1. 타일 수 계산:
   step_raw = 90° * (1 - 0.5) = 45°
   N = ceil(360° / 45°) = 8개

2. 균일 간격 재조정:
   step = 360° / 8 = 45° (정확히 닫힘)

3. 중심 배치 (start=-180°, seam_phase=0°):
   -180°, -135°, -90°, -45°, 0°, 45°, 90°, 135°
   (첫 중심과 마지막 중심이 정확히 360° 차이)
```

**장점**:
- 360° 완전 커버리지 보장
- 경계 누락 없음 (seam 처리 완벽)
- 균일한 타일 분포

**seam_phase_deg 활용**:
```python
# -180° 중심에 타일 배치하려면:
seam_phase_deg = -hfov_deg / 2  # -45°
→ 첫 중심: -180° + (-45°) = -225° = +135° (정규화)
→ 결과: -180°가 타일 중심이 됨
```

### 경계 중심 강제 추가
#### `maybe_add_seam_center(centers, seam_center=-180.0)`
- **목적**: -180°(= +180°) 위치에 타일 중심 명시적 추가
- **조건**: 기존 중심 중 ±1e-6° 이내에 없을 때만 추가

### 수직 방향 타일링
#### `make_pitch_centers(vfov_deg, overlap, pitch_min, pitch_max)`
```
예: vFOV=90°, overlap=0.5, pitch_min=-45°, pitch_max=45°

step = 90° * (1 - 0.5) = 45°
시작: -45° + 45°/2 = -22.5°
중심: -22.5°, 22.5°
```

---

## 4. ERP → Pinhole 변환

### `erp_to_pinhole_tile(erp, yaw_deg, pitch_deg, hfov_deg, out_size)`

**동작 과정**:
```
1. 입력: ERP 이미지 (numpy array, RGB)
2. RGB → BGR 변환 (py360convert 요구사항)
3. py360convert.e2p() 호출:
   - 중심: (yaw_deg, pitch_deg)
   - FOV: hfov_deg × vfov_deg
   - 출력: (out_size, out_size) pinhole 이미지
4. BGR → RGB 변환
5. 반환: PIL Image
```

**py360convert 내부 원리**:
- 출력 픽셀마다 카메라 광선 방향 계산
- 구면 좌표로 변환하여 ERP 이미지에서 샘플링
- Bilinear interpolation으로 부드러운 결과

---

## 5. 최종 통합: `build_anyres_from_erp()`

### 입력 파라미터
```python
erp_img: Image.Image              # 입력 ERP 파노라마
base_size: int = 336              # 글로벌 이미지 크기
tile_render_size: int = 672       # 타일 렌더링 해상도 (내부)
vit_size: Optional[int] = None    # 최종 ViT 입력 크기 (None=렌더 크기 유지)
hfov_deg: float = 90.0            # 타일 수평 FOV
overlap: float = 0.2              # 타일 간 겹침 비율
closed_loop_yaw: bool = False     # 폐곡선 분할 활성화 (권장: True)
yaw_phase_deg: float = 0.0        # 수평 위상 쉬프트
include_seam_center: bool = False # -180° 중심 강제 추가
pitch_min: float = -45.0          # 수직 최소 각도
pitch_max: float = 45.0           # 수직 최대 각도
pitch_full_span: bool = False     # -90°~+90° 전체 커버
cap_eps: float = 0.5              # ±90° 근처 안정화 여유
```

### 실행 단계

#### Step 1: 글로벌 이미지 생성
```python
# ERP 전체를 축소하여 정사각형으로
global_img = letterbox_square(erp_img, base_size=336)
# PIL → Tensor (3, 336, 336)
g_tensor = pil_to_tensor(global_img)
# ViT 크기로 리사이즈 (필요 시)
g_tensor = resize_to_vit(g_tensor, vit_size=336)
```

**역할**: 전체 파노라마 컨텍스트 제공 (저해상도)

#### Step 2: 타일 중심 계산
```python
# pitch_full_span=True면 수직 범위 자동 확장
if pitch_full_span:
    pitch_min = -90.0 + 0.5  # -89.5°
    pitch_max =  90.0 - 0.5  # +89.5°

# 수평 중심 리스트
if closed_loop_yaw:
    yaws = make_yaw_centers_closed_loop(90.0, 0.2)
    # 예: [-180°, -108°, -36°, 36°, 108°]
else:
    yaws = make_yaw_centers_standard(90.0, 0.2)

# -180° 중심 강제 추가 (옵션)
if include_seam_center:
    yaws = maybe_add_seam_center(yaws)

# 수직 중심 리스트
pitches = make_pitch_centers(90.0, 0.2, -45.0, 45.0)
# 예: [-22.5°, 22.5°]
```

#### Step 3: 타일 생성 (이중 루프)
```python
tiles_tensors = []
metas = []
tid = 0

for pitch in pitches:
    for yaw in yaws:
        # ERP → Pinhole 변환 (672×672)
        tile = erp_to_pinhole_tile(erp_np, yaw, pitch, hfov_deg=90.0, out_size=672)
        
        # PIL → Tensor (3, 672, 672)
        t_tensor = pil_to_tensor(tile)
        
        # ViT 크기로 리사이즈 (672 → 336)
        t_tensor = resize_to_vit(t_tensor, vit_size=336)
        
        tiles_tensors.append(t_tensor)
        
        # 메타데이터 기록
        metas.append(TileMeta(
            tile_id=tid,
            yaw_deg=yaw,
            pitch_deg=pitch,
            hfov_deg=90.0,
            vfov_deg=90.0,
            center_xyz=yaw_pitch_to_xyz(yaw, pitch)
        ))
        tid += 1
```

**생성 순서**: Pitch 우선, Yaw 내부 루프
```
Pitch 0: [Yaw 0], [Yaw 1], [Yaw 2], ...
Pitch 1: [Yaw 0], [Yaw 1], [Yaw 2], ...
```

#### Step 4: 최종 패킹
```python
# 타일 스택: (N, 3, vit_size, vit_size)
tiles = torch.stack(tiles_tensors, dim=0)

# 글로벌 메타데이터
gmeta = {
    "kind": "global_letterbox",
    "base_size": 336,
    "vit_size": 336,
    "note": "ERP 전체 컨텍스트(정사각 패킹)"
}

return AnyResPack(
    global_image=g_tensor,   # (3, 336, 336)
    tiles=tiles,             # (N, 3, 336, 336)
    metas=metas,             # List[TileMeta] × N
    global_meta=gmeta        # Dict
)
```

---

## 출력 데이터 구조

### `AnyResPack`
```python
@dataclass
class AnyResPack:
    global_image: torch.Tensor    # (3, G, G) - 글로벌 컨텍스트
    tiles: torch.Tensor           # (N, 3, T, T) - 로컬 타일들
    metas: List[TileMeta]         # N개 타일의 메타데이터
    global_meta: Dict             # 글로벌 이미지 정보
```

### `TileMeta`
```python
@dataclass
class TileMeta:
    tile_id: int                  # 0부터 시작하는 타일 ID
    yaw_deg: float                # 수평 중심 각도 [-180, 180)
    pitch_deg: float              # 수직 중심 각도 [-90, 90]
    hfov_deg: float               # 수평 FOV (예: 90.0)
    vfov_deg: float               # 수직 FOV (예: 90.0)
    center_xyz: Tuple[float, float, float]  # 3D 단위벡터
```

---

## 실제 사용 예시

### 기본 사용 (열린 구간)
```python
from PIL import Image
from anyres_e2p import build_anyres_from_erp

erp = Image.open("pano.jpg")
pack = build_anyres_from_erp(
    erp_img=erp,
    base_size=336,
    tile_render_size=672,
    vit_size=336,
    hfov_deg=90.0,
    overlap=0.5
)

print(f"글로벌: {pack.global_image.shape}")  # (3, 336, 336)
print(f"타일 수: {pack.tiles.shape[0]}")     # 예: 14개 (2 pitch × 7 yaw)
print(f"타일 크기: {pack.tiles.shape[1:]}")  # (3, 336, 336)
```

### 권장 설정 (폐곡선, 전체 커버)
```python
pack = build_anyres_from_erp(
    erp_img=erp,
    base_size=336,
    tile_render_size=672,
    vit_size=336,
    hfov_deg=90.0,
    overlap=0.5,
    closed_loop_yaw=True,        # 360° 완전 커버
    yaw_phase_deg=-45.0,         # -180° 중심 포함
    pitch_full_span=True,        # -90°~+90° 커버
    cap_eps=0.5
)

# 타일 수 계산:
# Yaw: ceil(360/45) = 8개
# Pitch: ceil(180/45) = 4개
# 총: 8 × 4 = 32개
```

### 메타데이터 활용
```python
for meta in pack.metas[:3]:
    print(f"타일 {meta.tile_id}: "
          f"yaw={meta.yaw_deg:+.1f}°, "
          f"pitch={meta.pitch_deg:+.1f}°, "
          f"xyz={meta.center_xyz}")

# 출력 예:
# 타일 0: yaw=-180.0°, pitch=-67.5°, xyz=(-1.0, -0.92, 0.0)
# 타일 1: yaw=-135.0°, pitch=-67.5°, xyz=(-0.71, -0.92, 0.71)
# 타일 2: yaw=-90.0°, pitch=-67.5°, xyz=(0.0, -0.92, 1.0)
```

---

## 파라미터 튜닝 가이드

### 1. `overlap` (타일 겹침 비율)
| 값 | 타일 수 | 장점 | 단점 |
|---|---|---|---|
| 0.0 | 최소 (8 yaw × 2 pitch = 16) | 빠른 처리 | 경계 정보 손실 |
| 0.5 | 중간 (8 × 4 = 32) | 균형 잡힌 커버리지 | - |
| 0.8 | 최대 (36 × 9 = 324) | 풍부한 컨텍스트 | 메모리/속도 문제 |

**권장**: `0.3 ~ 0.5` (30~50% 겹침)

### 2. `hfov_deg` (타일 FOV)
| 값 | 타일 수 | 해상도 밀도 | 특징 |
|---|---|---|---|
| 60° | 많음 | 높음 | 디테일 중시 (작은 객체 탐지) |
| 90° | 중간 | 중간 | 균형 (기본 권장) |
| 120° | 적음 | 낮음 | 컨텍스트 중시 (넓은 시야) |

**권장**: `90°` (인간 시야와 유사)

### 3. `tile_render_size` vs `vit_size`
```python
# 고품질 렌더링 후 다운샘플링 (권장)
tile_render_size=672, vit_size=336
→ 고해상도 렌더링 후 안티앨리어싱 효과

# 직접 렌더링 (빠름)
tile_render_size=336, vit_size=336
→ 속도 우선, 약간의 품질 저하

# 초고해상도 (연구용)
tile_render_size=1344, vit_size=336
→ 최고 품질, 매우 느림
```

### 4. `pitch_min` / `pitch_max`
| 설정 | 적용 시나리오 |
|---|---|
| `-45° ~ +45°` | 실내 (천장/바닥 덜 중요) |
| `-60° ~ +60°` | 일반 실외 |
| `-89.5° ~ +89.5°` (`pitch_full_span=True`) | 하늘/땅 중요 (드론 촬영 등) |

---

## 주요 디자인 결정

### 왜 폐곡선 방식을 권장하는가?
1. **완전 커버리지**: 360° 경계(seam) 누락 없음
2. **균일 분포**: 모든 타일이 동일한 간격
3. **수치 안정성**: 정확히 360°로 닫힘

### 왜 글로벌 이미지를 letterbox로 만드는가?
- ERP는 2:1 비율 (예: 4096×2048)
- ViT는 정사각형 입력 요구 (예: 336×336)
- Letterbox: 종횡비 유지하며 정사각형 패딩
- 대안 (crop): 정보 손실 발생

### 왜 타일을 고해상도로 렌더링 후 다운샘플링?
- `py360convert`의 보간 품질 향상
- 안티앨리어싱 효과 (모아레 패턴 감소)
- 타일 경계의 아티팩트 완화

---

## 제약사항 및 알려진 이슈

### 1. py360convert 의존성
- OpenCV 필요 (`opencv-python`)
- CPU 기반 처리 (GPU 가속 없음)
- 대용량 이미지 시 느림

### 2. 메모리 사용량
```
단일 샘플 메모리 = global + tiles
= (3 × 336 × 336) + (N × 3 × 336 × 336)
= 0.33MB + N × 0.33MB

예: N=32 타일 → 약 11MB (float32)
배치 크기 16 → 176MB (타일만)
```

### 3. 극점(±90°) 처리
- `cap_eps=0.5`로 ±90° 정확히 피함
- 이유: `py360convert`가 극점에서 불안정
- 해결: Cube map 방식으로 대체 가능

### 4. 타일 순서 의존성
- **현재**: Pitch 우선 순회 (pitch 0 → all yaws, pitch 1 → all yaws, ...)
- **주의**: 코드 변경 시 `tile_id` 일관성 유지 필요
- **PE 계산**: 타일 순서와 메타데이터 정렬 필수

---

## 다음 단계 (코드에 포함되지 않음)

### 1. 포지셔널 임베딩 (PE) 추가
```python
# metas의 center_xyz 활용
def add_positional_embedding(tiles, metas, embed_dim):
    pe = []
    for meta in metas:
        x, y, z = meta.center_xyz
        pe.append(mlp([x, y, z]))  # (embed_dim,)
    pe = torch.stack(pe, dim=0)  # (N, embed_dim)
    return tiles + pe.unsqueeze(1).unsqueeze(2)
```

### 2. Vision Encoder 통합
```python
# VisionBackbone으로 타일 인코딩
global_feat = vision_encoder(pack.global_image.unsqueeze(0))  # (1, D)
tile_feats = vision_encoder(pack.tiles)  # (N, D)

# Resampler로 통합
fused = resampler(torch.cat([global_feat, tile_feats], dim=0))
```

### 3. VICReg Loss 적용
```python
# 겹치는 타일 간 feature 일관성 학습
for i, meta_i in enumerate(metas):
    for j, meta_j in enumerate(metas):
        if is_overlapping(meta_i, meta_j):
            loss += vicreg_loss(tile_feats[i], tile_feats[j])
```

---

## 관련 문서
- **ANYRES_ERP_INTEGRATION.md**: PanoLLaVA 파이프라인 통합 가이드
- **CONFIG_GUIDE.md**: `crop_strategy: anyres_e2p` 설정 방법
- **VLM_MODEL_UPDATES.md**: Vision encoder와의 연동 방식

---

## 참고 자료
- [py360convert 문서](https://github.com/sunset1995/py360convert)
- [LLaVA-NeXT AnyRes 논문](https://arxiv.org/abs/2310.03744)
- [Equirectangular Projection 설명](https://en.wikipedia.org/wiki/Equirectangular_projection)
