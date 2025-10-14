# 평가 스크립트 사용 및 메트릭 통일 가이드

이 문서는 프로젝트의 세 가지 평가 스크립트의 사용처와 평가 지표 통일 작업을 설명합니다.

## 📊 평가 스크립트 개요

### 1. scripts/eval.py (메인 PanoramaVLM 평가)

**목적**: 학습된 PanoramaVLM 모델을 평가하는 메인 스크립트

**사용처**:
- 학습 완료 후 모델 성능 측정
- `scripts/train.sh`에서 참조됨
- 설정 파일 기반 자동 모델 디렉토리 탐색

**평가 지표**:
- ✅ BLEU-4 (corpus-level, smoothing 적용)
- ✅ METEOR (wordnet 기반 동의어 매칭)
- ✅ ROUGE-L (최장 공통 부분수열)
- ✅ SPICE (타임아웃 처리 + 의미 유사도 대안)
- ✅ CIDEr (consensus-based 이미지 설명 평가)

**사용 예시**:
```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --csv-input data/quic360/test.csv
```

**특징**:
- 모델 디렉토리 자동 탐색 (`resolve_model_dir`)
- LoRA 가중치 자동 로드
- HF 모델/체크포인트 모두 지원
- 가장 완전하고 안정적인 메트릭 구현

---

### 2. scripts/evaluate_vlm_models.py (HF VLM 모델 비교)

**목적**: HuggingFace의 다양한 VLM 모델들을 동일 데이터셋으로 비교 평가

**사용처**:
- `scripts/run_vlm_ablation.sh` (ablation study)
- `scripts/test_vlm_eval.sh` (빠른 테스트)
- `docs/VLM_EVALUATION_GUIDE.md`에 문서화됨

**지원 모델**:
- LLaVA 1.5/1.6
- BLIP2 (OPT/Flan-T5)
- InstructBLIP
- Qwen2.5-VL (3B)
- InternVL2 (2B)
- Gemma 3 (4B)

**평가 지표** (수정 후):
- ✅ **eval.py의 calculate_evaluation_metrics 재사용** (우선)
- ✅ 로컬 구현 폴백 (호환성 유지)
- ✅ 완전히 동일한 메트릭 보장

**사용 예시**:
```bash
# 여러 모델 비교
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b blip2-opt-2.7b qwen2-vl-2b \
    --output_dir results \
    --batch_size 2

# 단일 모델 평가
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models internvl2-2b \
    --max_samples 50
```

**변경 사항** (2025-01-XX):
- ✅ `scripts.eval` 모듈에서 `calculate_evaluation_metrics` 임포트
- ✅ `compute_text_metrics` 함수가 eval.py 구현 우선 사용
- ✅ 실패 시 로컬 구현으로 자동 폴백
- ✅ 로그에 사용된 구현 명시

**출력**:
```
results/ablation/{model_name}/
├── metrics.json          # 평가 메트릭 (BLEU, METEOR, ROUGE, SPICE, CIDEr)
└── predictions.csv       # 예측/정답 비교
```

---

### 3. scripts/vlm_evaluate.py (LoRA 튜닝 VLM 평가)

**목적**: LoRA 어댑터가 적용된 VLM 모델 평가

**사용처**:
- `results/vlm_lora_ablation/` 디렉토리의 LoRA 실험 평가
- 자동 어댑터 탐색 (lora_adapter/, final/, checkpoints/)

**평가 지표**:
- ✅ **eval.py의 함수들을 직접 재사용** (이미 구현됨)
  ```python
  from scripts.eval import (
      calculate_evaluation_metrics,
      save_and_log_results,
  )
  ```

**사용 예시**:
```bash
# LoRA 실험 평가 (자동 탐색)
python scripts/vlm_evaluate.py \
    --csv data/quic360/test.csv \
    --run qwen_vl_chat__lora_r16 \
    --results-root results/vlm_lora_ablation

# 명시적 모델 지정
python scripts/vlm_evaluate.py \
    --csv data/quic360/test.csv \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --lora-path results/vlm_lora_ablation/qwen2_vl__r8/lora_adapter
```

**특징**:
- ✅ 처음부터 eval.py 재사용으로 설계됨
- ✅ 메트릭 일관성 보장
- ✅ 어댑터 자동 탐색
- ✅ 베이스 모델 정보 자동 추출

---

## 🔄 메트릭 통일 작업 요약

### 변경 전 문제점
- `evaluate_vlm_models.py`가 메트릭 계산 로직을 **재구현**
- eval.py와 미묘한 차이 가능성
- 유지보수 중복

### 변경 후 해결
1. **evaluate_vlm_models.py 수정**:
   - `scripts.eval.calculate_evaluation_metrics` 임포트
   - `compute_text_metrics` 함수 내부에서 eval.py 구현 우선 사용
   - 임포트 실패 시 로컬 구현으로 폴백 (하위 호환성)

2. **vlm_evaluate.py**:
   - 이미 올바르게 구현됨 (수정 불필요)

### 메트릭 계산 흐름

```
모든 평가 스크립트
        ↓
scripts/eval.py::calculate_evaluation_metrics()
        ↓
    ┌───────────────────────────────────┐
    │  BLEU-4   (corpus_bleu)          │
    │  METEOR   (meteor_score)          │
    │  ROUGE-L  (rouge_scorer)          │
    │  SPICE    (타임아웃 처리 + 대안)   │
    │  CIDEr    (cider_scorer)          │
    └───────────────────────────────────┘
```

---

## 📈 평가 지표 상세

### BLEU-4
- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **설명**: 4-gram precision with smoothing
- **용도**: 기계 번역/생성 품질 측정

### METEOR
- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **설명**: 동의어/어형 변화 고려한 F1 점수
- **용도**: BLEU보다 인간 판단과 높은 상관관계

### ROUGE-L
- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **설명**: 최장 공통 부분수열 기반 F1
- **용도**: 요약 품질 평가

### SPICE
- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **설명**: 의미 그래프 기반 이미지 캡션 평가
- **용도**: 의미적 정확성 측정
- **특징**: eval.py는 타임아웃 처리 + 대안 구현 포함

### CIDEr
- **범위**: 0.0 ~ 10.0 (높을수록 좋음)
- **설명**: Consensus-based 이미지 설명 평가
- **용도**: 여러 정답과의 일치도

---

## 🧪 테스트 및 검증

### 빠른 테스트
```bash
# evaluate_vlm_models.py 테스트
bash scripts/test_vlm_eval.sh

# 결과 확인
cat eval_results/test_run/ablation/blip2-opt-2.7b/metrics.json
```

### 전체 ablation study
```bash
bash scripts/run_vlm_ablation.sh
```

### 결과 비교
```bash
# 모든 모델의 메트릭 비교
python -c "
import json
from pathlib import Path

results_dir = Path('results/ablation')
for model_dir in results_dir.iterdir():
    if model_dir.is_dir():
        metrics_file = model_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            print(f'{model_dir.name}:')
            for k, v in metrics.items():
                print(f'  {k}: {v:.4f}')
"
```

---

## 📝 권장 사항

1. **새로운 평가 스크립트 작성 시**:
   - 항상 `scripts.eval.calculate_evaluation_metrics` 재사용
   - DataFrame 형식으로 예측/정답 전달
   ```python
   from scripts.eval import calculate_evaluation_metrics

   df = pd.DataFrame({
       'prediction': predictions,
       'reference': references,
   })

   metrics = calculate_evaluation_metrics(
       df,
       output_dir=Path('eval_results'),
       timestamp='20250101_120000',
       prefix='my_model'
   )
   ```

2. **메트릭 추가/수정 시**:
   - `scripts/eval.py`에서만 수정
   - 다른 스크립트는 자동으로 새 메트릭 사용

3. **디버깅**:
   - 로그에서 "✓ Using shared evaluation metrics from scripts/eval.py" 확인
   - 폴백 경고 발생 시 임포트 경로 점검

---

## 🔗 관련 문서

- [VLM_EVALUATION_GUIDE.md](./VLM_EVALUATION_GUIDE.md): HF VLM 모델 평가 상세 가이드
- [IMPROVED_USAGE.md](./IMPROVED_USAGE.md): 전체 학습/평가 파이프라인

---

## 변경 이력

- **2025-01-XX**: `evaluate_vlm_models.py` 메트릭 통일 작업 완료
- **2025-01-XX**: 문서 작성

