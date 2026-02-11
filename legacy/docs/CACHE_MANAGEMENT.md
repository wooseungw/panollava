# 캐시 관리 및 학습 안정성 개선

## 문제점 분석

학습 중 프로그램이 중단되는 주요 원인:

1. **디스크 공간 부족**
   - HuggingFace 캐시: 203GB
   - Runs 디렉토리: 50GB
   - 전체 디스크 사용률: 87%

2. **손상된 이미지 파일**
   - Structured3D 데이터셋에 손상된 PNG 파일 다수 존재
   - 에러: `broken PNG file (chunk b'\x00\x00\x00\x00')`
   - 재귀 호출로 인한 무한 루프 가능성

3. **메모리 누수**
   - 장시간 학습 시 GPU 캐시 누적
   - 정기적인 캐시 정리 메커니즘 부재

## 해결 방안

### 1. HuggingFace 캐시 최적화

[scripts/train.py](../scripts/train.py)에 환경 변수 추가:

```python
# HuggingFace 캐시 최적화 설정 (메모리 절약)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
```

### 2. 손상된 이미지 처리 개선

[src/panovlm/dataset.py](../src/panovlm/dataset.py)의 `_load_image` 함수 개선:

**변경 전 (문제점)**:
```python
def _load_image(self, image_path: str, idx: int) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        # 무한 재귀 가능성!
        return self._load_image(self.df.iloc[(idx + 1) % len(self)].url, (idx + 1) % len(self))
```

**변경 후 (개선)**:
```python
def _load_image(self, image_path: str, idx: int, max_retries: int = 10) -> Image.Image:
    """이미지 로드 및 에러 처리 (손상된 파일 자동 스킵)"""
    retries = 0
    current_idx = idx

    while retries < max_retries:
        try:
            current_path = self.df.iloc[current_idx].url if retries > 0 else image_path
            # PIL이 파일을 완전히 읽도록 강제 (손상된 청크 감지)
            img = Image.open(current_path)
            img.load()  # 이미지 데이터를 메모리에 완전히 로드
            return img.convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {current_path}: {e} (retry {retries + 1}/{max_retries})")
            retries += 1
            current_idx = (current_idx + 1) % len(self)

    # 모든 재시도 실패 시 검은색 이미지 반환 (학습 중단 방지)
    logger.error(f"All retries failed for idx {idx}, returning black placeholder image")
    return Image.new("RGB", (224, 224), (0, 0, 0))
```

**개선 사항**:
- ✅ 재귀 호출 제거 → 스택 오버플로 방지
- ✅ `img.load()` 추가 → 손상된 청크 조기 감지
- ✅ 최대 재시도 횟수 제한 (10회)
- ✅ 실패 시 placeholder 이미지 반환 → 학습 중단 방지

### 3. 주기적 캐시 정리

[configs/default.yaml](../configs/default.yaml)에 설정 추가:

```yaml
training:
  # 캐시 정리 설정 (메모리 관리)
  cache_cleanup_interval: 1000  # N 스텝마다 캐시 정리 (0=비활성화)
```

[scripts/train.py](../scripts/train.py) training_step에서 자동 정리:

```python
# 주기적 캐시 정리 (메모리 누수 방지) - 설정 가능
if self.cache_cleanup_interval > 0 and batch_idx % self.cache_cleanup_interval == 0 and torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

**권장 설정**:
- 일반적인 경우: `1000` (기본값)
- 대용량 배치: `500`
- 소형 모델: `2000` 또는 `0` (비활성화)

### 4. 캐시 정리 스크립트

수동 캐시 정리를 위한 스크립트 제공: [scripts/cleanup_cache.sh](../scripts/cleanup_cache.sh)

```bash
# 실행 방법
bash scripts/cleanup_cache.sh
```

**기능**:
1. Python `__pycache__` 정리
2. HuggingFace 캐시 임시 파일 정리
3. 오래된 체크포인트 삭제 (30일+)
4. WandB 로그 정리
5. 손상된 PNG 파일 검사 및 삭제

## 사용 가이드

### 학습 전 준비

```bash
# 1. 디스크 공간 확인
df -h /data

# 2. 캐시 정리 (필요시)
bash scripts/cleanup_cache.sh

# 3. 학습 시작
python scripts/train.py --config configs/default.yaml
```

### 설정 커스터마이징

`configs/default.yaml` 수정:

```yaml
training:
  # 메모리가 부족한 경우 더 자주 정리
  cache_cleanup_interval: 500

  # 또는 성능 최우선인 경우 비활성화
  cache_cleanup_interval: 0
```

### 손상된 이미지 사전 제거

```bash
# 손상된 파일 찾기
bash scripts/cleanup_cache.sh
# → 5번 옵션 선택

# 또는 Python으로 직접 검사
python -c "
from PIL import Image
import sys
for path in sys.argv[1:]:
    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        print(f'Broken: {path}')
" /data/path/to/*.png
```

## 모니터링

### 학습 중 디스크 사용량 모니터링

```bash
# 별도 터미널에서 실행
watch -n 10 'df -h /data && echo && du -sh ~/.cache/huggingface runs'
```

### 로그 확인

```bash
# 손상된 이미지 로그 확인
grep "Failed to load image" training.log | tail -n 20

# 캐시 정리 로그 확인
grep "empty_cache\|gc.collect" training.log
```

## 예상 효과

1. **안정성 향상**
   - 손상된 이미지로 인한 중단 방지
   - OOM 에러 감소

2. **디스크 공간 절약**
   - 임시 파일 자동 정리
   - 오래된 체크포인트 관리

3. **성능 유지**
   - 적절한 캐시 정리 주기 설정
   - GPU 메모리 효율화

## 문제 해결

### 여전히 학습이 중단되는 경우

1. **디스크 공간 재확인**
   ```bash
   df -h /data
   # 90% 이상이면 캐시 정리 필수
   ```

2. **배치 크기 감소**
   ```yaml
   training:
     stage_configs:
       vision:
         batch_size: 8  # 16 → 8로 감소
   ```

3. **num_workers 감소**
   ```yaml
   training:
     num_workers: 8  # 16 → 8로 감소
   ```

4. **더 자주 캐시 정리**
   ```yaml
   training:
     cache_cleanup_interval: 500  # 1000 → 500
   ```

## 참고 자료

- [PyTorch 메모리 관리](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [HuggingFace 캐시 관리](https://huggingface.co/docs/datasets/cache)
- [PIL 이미지 로딩](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)
