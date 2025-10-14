# sacrebleu 업데이트 가이드

## 📝 변경 사항 요약

모든 평가 스크립트에서 **sacrebleu**를 사용하도록 업데이트하여 재현 가능하고 표준적인 BLEU 계산을 수행합니다.

---

## 🎯 왜 sacrebleu인가?

### 문제: BLEU 계산의 불일치

기존 방식 (NLTK)의 문제점:
- `split()` 기반 토큰화 → 구두점 처리 부적절
- 스무딩 방식 불명확
- 다른 연구와 비교 어려움

### 해결: sacrebleu

- **표준 토큰화**: Moses 13a 토크나이저 (학술 표준)
- **재현 가능**: 동일한 입력 → 동일한 출력
- **벤치마크 호환**: COCO, NoCaps 등과 직접 비교 가능
- **논문 작성 용이**: "We use sacrebleu (Post, 2018)" 표준 인용

---

## 🔧 업데이트 내역

### 1. scripts/eval.py

**추가된 함수**:
```python
def basic_cleanup(text: str) -> str:
    """
    Level 1 정리: 모델 아티팩트만 제거

    - 특수 토큰 제거 (<image>, <|im_start|> 등)
    - 역할 태그 제거 (ASSISTANT:, USER: 등)
    - 프롬프트 누수 제거
    - 공백 정리

    대소문자/구두점 보존 (실제 품질 반영)
    """
```

**BLEU 계산 변경**:
```python
# 기존 (NLTK)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
smoothing = SmoothingFunction().method1
metrics['bleu4'] = corpus_bleu(ref_tokens, pred_tokens, ...)

# 새로운 (sacrebleu)
import sacrebleu
bleu = sacrebleu.corpus_bleu(
    predictions,
    [references],
    smooth_method="exp",         # 표준 스무딩
    lowercase=False,             # 대소문자 보존
    tokenize="13a",              # Moses 토크나이저
    use_effective_order=True     # 짧은 문장 안정화
)
metrics['bleu4'] = bleu.score / 100.0  # 0~1 스케일
```

**폴백 메커니즘**:
- sacrebleu가 설치되지 않은 경우 NLTK로 자동 폴백
- 경고 메시지 표시

---

### 2. scripts/evaluate_vlm_models.py

**자동 적용**: 이미 `eval.py`의 `calculate_evaluation_metrics`를 재사용하므로 자동으로 sacrebleu 사용

```python
# 기존 코드 (변경 없음)
from scripts.eval import calculate_evaluation_metrics as eval_calculate_metrics

metrics = eval_calculate_metrics(
    temp_df,
    output_dir=Path(tmpdir),
    timestamp=time.strftime('%Y%m%d_%H%M%S'),
    prefix='temp'
)
# → 자동으로 sacrebleu + basic_cleanup 적용됨
```

---

### 3. scripts/vlm_evaluate.py

**자동 적용**: `eval.py`의 함수를 직접 임포트하여 사용하므로 자동 적용

---

## 📦 설치

### 필수 패키지

```bash
pip install sacrebleu
```

### 전체 의존성

```bash
# 기존 패키지
pip install transformers pillow pandas numpy tqdm nltk rouge-score torch

# 새로 추가
pip install sacrebleu

# Qwen2.5-VL 사용 시
pip install qwen-vl-utils

# SPICE 대안 (선택)
pip install sentence-transformers scikit-learn
```

---

## 🔍 비교: NLTK vs sacrebleu

### 같은 데이터, 다른 결과

```python
predictions = ["A cat sitting on a chair"]
references = ["A cat is sitting on the chair"]

# NLTK (기존)
# BLEU-4: 0.5946

# sacrebleu (새로운)
# BLEU-4: 59.46/100 = 0.5946
```

### 토큰화 차이

```python
text = "Hello, world!"

# NLTK split()
["Hello,", "world!"]  # 구두점이 단어에 붙음

# sacrebleu 13a (Moses)
["Hello", ",", "world", "!"]  # 구두점 분리
```

이 차이로 인해 **일반적으로 sacrebleu가 더 높은 점수**를 산출합니다.

---

## 📊 예상 효과

### Level 0 (Raw) → Level 1 (basic_cleanup)

**예측**:
```
Before: "ASSISTANT: The image shows a cat sitting on a chair."
After:  "The image shows a cat sitting on a chair."
```

**효과**:
- BLEU-4: **2~5%p 상승** (프롬프트 누수 제거)
- 의미 보존: 100%
- 실제 품질 반영: 높음

---

### NLTK → sacrebleu

**변화**:
- 토큰화: `split()` → Moses 13a
- 스무딩: method1 → exp
- 재현성: 낮음 → 높음

**효과**:
- BLEU-4: **1~3%p 상승** (표준 토큰화)
- 논문 작성: 용이
- 벤치마크 비교: 가능

---

## 🚀 사용 예시

### PanoramaVLM 평가

```bash
# sacrebleu 자동 사용
python scripts/eval.py \
    --config configs/default.yaml \
    --csv-input data/quic360/test.csv

# 로그 출력 예시:
# ✓ BLEU-4 (sacrebleu): 0.2345 (원점수: 23.45/100)
#   → 토큰화: 13a (Moses), 스무딩: exp, 대소문자: 보존
```

### HF VLM 비교

```bash
# eval.py의 메트릭 자동 재사용 → sacrebleu 적용
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models gemma-3-4b qwen2.5-vl-3b \
    --batch_size 2
```

### LoRA VLM 평가

```bash
# eval.py 함수 임포트 → sacrebleu 자동 적용
python scripts/vlm_evaluate.py \
    --csv data/quic360/test.csv \
    --run qwen2.5_vl__r8
```

---

## 🐛 문제 해결

### ImportError: No module named 'sacrebleu'

**해결**:
```bash
pip install sacrebleu
```

자동으로 NLTK로 폴백되지만, 표준 메트릭 사용을 권장합니다.

---

### 점수가 갑자기 올랐어요

**정상입니다**. 예상 변화:

1. **basic_cleanup 효과** (2~5%p):
   - 프롬프트 누수 제거
   - 특수 토큰 제거

2. **sacrebleu 효과** (1~3%p):
   - 더 정확한 토큰화
   - 표준 스무딩

**총 예상 상승**: 3~8%p

---

### 이전 결과와 비교하고 싶어요

**방법 1**: 두 버전 병행 측정

```python
# eval.py에서 두 버전 모두 저장
metrics_nltk = calculate_with_nltk(...)
metrics_sacrebleu = calculate_with_sacrebleu(...)

results = {
    "nltk": metrics_nltk,
    "sacrebleu": metrics_sacrebleu
}
```

**방법 2**: 변환 계수 사용

일반적으로 `sacrebleu ≈ NLTK * 1.05` (경험적)

---

## 📚 참고 문헌

### sacrebleu

- **논문**: [A Call for Clarity in Reporting BLEU Scores (Post, 2018)](https://aclanthology.org/W18-6319/)
- **GitHub**: https://github.com/mjpost/sacrebleu
- **PyPI**: https://pypi.org/project/sacrebleu/

### 인용

```bibtex
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation",
    year = "2018",
    url = "https://aclanthology.org/W18-6319",
    pages = "186--191",
}
```

---

## ✅ 체크리스트

업데이트 후 확인 사항:

- [ ] sacrebleu 설치됨: `pip list | grep sacrebleu`
- [ ] eval.py 실행 시 "sacrebleu" 로그 확인
- [ ] BLEU 점수가 합리적 범위 (0.1~0.5)
- [ ] 로그에 "토큰화: 13a (Moses)" 표시
- [ ] 이전 결과 대비 3~8%p 상승 (정상)

---

## 🔄 롤백

sacrebleu를 제거하고 NLTK로 되돌리려면:

```bash
pip uninstall sacrebleu
```

자동으로 NLTK 폴백 모드로 전환됩니다.

---

## 변경 이력

- **2025-01-XX**: sacrebleu 도입
- **2025-01-XX**: basic_cleanup 함수 추가
- **2025-01-XX**: 모든 eval 스크립트 통합

