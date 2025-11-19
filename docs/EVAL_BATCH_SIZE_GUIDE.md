# Evaluation Configuration Guide

## 🎯 주요 설정 옵션

### 1. CSV 입력 파일 (Test Dataset)

**우선순위:**
```
1. CLI --csv-input
   ↓
2. config.data.csv_test
   ↓
3. config.data.csv_val
   ↓
4. config.paths.csv_val
   ↓
5. 기본값: "data/quic360/test.csv"
```

**설정 방법:**

**CLI:**
```bash
python scripts/eval.py \
    --checkpoint-dir runs/.../finetune/... \
    --csv-input data/my_test.csv
```

**Config:**
```yaml
# configs/default.yaml
data:
  csv_test: "data/quic360/test.csv"  # Evaluation용 (기본값)
```

---

### 2. 배치 사이즈 설정

## 🎯 배치 사이즈 설정 방법 (3가지)

### 방법 1: CLI 인자로 지정 (최우선)

```bash
python scripts/eval.py \
    --checkpoint-dir runs/my_model/finetune/... \
    --batch-size 2
```

**장점:**
- ✅ 가장 간단하고 빠름
- ✅ Config 수정 불필요
- ✅ 일회성 실험에 적합

---

### 방법 2: Config 파일에서 지정

**`configs/default.yaml`** 또는 **`runs/.../config.yaml`**:
```yaml
training:
  eval_batch_size: 2  # 평가 배치 크기
  num_workers: 8
```

**장점:**
- ✅ 재현 가능
- ✅ 여러 실험에서 일관된 설정
- ✅ 문서화됨

**사용:**
```bash
# Config에 eval_batch_size 설정되어 있으면 자동 사용
python scripts/eval.py --checkpoint-dir runs/my_model/finetune/...
```

---

### 방법 3: 기본값 사용

아무것도 지정하지 않으면 **기본값 2** 사용

```bash
# eval_batch_size 없으면 자동으로 2 사용
python scripts/eval.py --checkpoint-dir runs/my_model/finetune/...
```

---

## 🔍 우선순위 (높은 것부터)

```
1. CLI --batch-size 인자
   ↓ (없으면)
2. config.training.eval_batch_size
   ↓ (없으면)
3. config.training.batch_size
   ↓ (없으면)
4. config.training.stage_configs.finetune.batch_size
   ↓ (없으면)
5. 기본값: 2
```

---

## 💡 배치 사이즈 선택 가이드

### GPU 메모리별 권장 크기

| GPU VRAM | 권장 Batch Size | 비고 |
|----------|----------------|------|
| 8GB | 1 | 안전 |
| 12GB | 2 | **기본값** ⭐ |
| 16GB | 4 | 효율적 |
| 24GB+ | 8-16 | 최대 성능 |

### AnyRes Strategy별 권장

| Crop Strategy | 권장 Batch Size | 이유 |
|---------------|----------------|------|
| `e2p` | 4-8 | 단일 view, 가벼움 |
| `anyres` | 2-4 | 중간 |
| `anyres_e2p` | 1-2 | 다중 tiles, 무거움 ⚠️ |
| `sliding_window` | 2-4 | 중간 |

### 모델 크기별 권장

| Language Model | 권장 Batch Size |
|----------------|----------------|
| Qwen3-0.6B | 4-8 |
| Qwen3-1.8B | 2-4 |
| Qwen3-7B | 1-2 |

---

## 📊 예시

### 작은 GPU (8GB)
```bash
python scripts/eval.py \
    --checkpoint-dir runs/my_model/finetune/... \
    --batch-size 1
```

### 일반적인 경우 (12-16GB)
```yaml
# configs/my_config.yaml
training:
  eval_batch_size: 2  # 기본값, config에 명시
```

```bash
python scripts/eval.py --checkpoint-dir runs/my_model/finetune/...
# → batch_size=2 자동 사용
```

### 대용량 GPU (24GB+)
```yaml
# configs/my_config.yaml
training:
  eval_batch_size: 8  # 큰 배치로 빠른 평가
```

---

## ⚠️ 주의사항

### 1. OOM (Out of Memory) 발생 시
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**해결:**
```bash
# 배치 크기를 절반으로 줄이기
python scripts/eval.py --checkpoint-dir ... --batch-size 1
```

### 2. 너무 큰 배치 사이즈
- **문제**: 메모리 초과, 느린 처리
- **증상**: GPU 메모리 99% 사용, swap 발생
- **권장**: GPU 메모리의 80% 이내 사용

### 3. 너무 작은 배치 사이즈
- **문제**: GPU 활용도 낮음, 느린 평가
- **증상**: GPU 메모리 30% 이하 사용
- **권장**: 가능한 한 크게 (OOM 직전까지)

---

## 🧪 최적 배치 크기 찾기

### 방법 1: Binary Search
```bash
# 시작: 8
python scripts/eval.py --checkpoint-dir ... --batch-size 8
# OOM 발생 → 4로 줄임
python scripts/eval.py --checkpoint-dir ... --batch-size 4
# 성공 → 6 시도
python scripts/eval.py --checkpoint-dir ... --batch-size 6
# 성공 → 최적값 6
```

### 방법 2: GPU 모니터링
```bash
# Terminal 1
watch -n 1 nvidia-smi

# Terminal 2
python scripts/eval.py --checkpoint-dir ... --batch-size 4
```

**목표**: GPU Memory 사용률 **70-85%**

---

## 📝 Config 예시

### 기본 설정 (권장)
```yaml
# configs/default.yaml
training:
  eval_batch_size: 2  # 대부분의 경우 안정적
  num_workers: 8
```

### 고성능 GPU
```yaml
# configs/high_performance.yaml
training:
  eval_batch_size: 8  # RTX 4090, A100 등
  num_workers: 16
```

### 저사양 GPU
```yaml
# configs/low_memory.yaml
training:
  eval_batch_size: 1  # RTX 3060, GTX 1080 등
  num_workers: 4
```

---

## 🔧 문제 해결

### Q: Config에 설정했는데 적용 안됨
```bash
# 확인: 로그에서 배치 크기 확인
python scripts/eval.py --checkpoint-dir ... 2>&1 | grep "배치 크기"
# 출력: "배치 크기: 2"
```

### Q: CLI 인자가 무시됨
```bash
# CLI 인자는 최우선순위 - 항상 적용됨
python scripts/eval.py --checkpoint-dir ... --batch-size 4
# → 무조건 4 사용
```

### Q: 어떤 값이 사용되는지 확인하고 싶음
```bash
# 평가 로그 시작 부분에 표시됨:
# ✓ 데이터셋 준비 완료
#    - 총 배치 수: 69
#    - 배치 크기: 2  ← 여기!
```

---

## 📚 관련 설정

배치 크기와 함께 조정하면 좋은 설정들:

```yaml
training:
  eval_batch_size: 2        # 평가 배치 크기
  num_workers: 8            # 데이터 로딩 워커 수
  
image_processing:
  anyres_max_patches: 9     # AnyRes 타일 개수 (메모리 영향)

generation:
  max_new_tokens: 128       # 생성 토큰 수 (메모리 영향)
```

---

## ✅ 요약

**간단 사용:**
```bash
# 방법 1: CLI로 지정
python scripts/eval.py --checkpoint-dir runs/.../finetune/... --batch-size 2

# 방법 2: Config에 설정 후 실행
python scripts/eval.py --checkpoint-dir runs/.../finetune/...
```

**권장 값:**
- 기본: `2` (12GB GPU)
- 작은 GPU: `1` (8GB)
- 큰 GPU: `4-8` (24GB+)

**우선순위:**
`CLI > config.eval_batch_size > config.batch_size > default(2)`

끝! 🎉
