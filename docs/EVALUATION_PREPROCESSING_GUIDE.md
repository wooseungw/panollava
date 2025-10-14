# 평가 전처리 가이드

## 🎯 목표

VLM 평가 시 **공정하고 일관된** 비교를 위한 전처리 전략을 수립합니다.

---

## 📋 전처리 레벨 정의

### Level 0: Raw (원본)
**적용**: 없음
**목적**: 모델의 실제 출력 그대로 평가
**사용 사례**:
- 프로덕션 환경 시뮬레이션
- 모델 간 순수 출력 품질 비교

```python
predictions = [model_output]  # 그대로
references = [ground_truth]    # 그대로
```

**장점**: 실제 사용 환경과 일치
**단점**: 토큰화/대소문자 차이로 메트릭 변동 큼

---

### Level 1: Basic Cleanup (기본 정리) ⭐ **권장**
**적용**:
- 특수 토큰 제거 (`<image>`, `<|im_start|>` 등)
- 역할 태그 제거 (`ASSISTANT:`, `USER:` 등)
- 프롬프트 누수 제거
- 과도한 공백 정리

```python
def basic_cleanup(text: str) -> str:
    """Level 1: 모델 아티팩트만 제거"""
    import re

    # 1. 특수 토큰 제거
    text = re.sub(r"<\|.*?\|>|<image>|</image>|<img>|</img>", " ", text, flags=re.I)
    text = re.sub(r"<vision_start>|<vision_end>|<image_pad>", " ", text, flags=re.I)

    # 2. 역할 태그 제거
    text = re.sub(r"^(USER:|ASSISTANT:|Question:|Answer:)\s*", "", text, flags=re.I)

    # 3. 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

**장점**:
- 모델 출력의 의미 보존
- 프롬프트 엔지니어링 아티팩트만 제거
- 대소문자, 구두점 등 원본 유지

**단점**:
- 토큰화 차이로 인한 BLEU 변동 여전히 존재

---

### Level 2: Standard Normalization (표준 정규화)
**적용**: Level 1 +
- 표준 토큰화 (Moses/sacrebleu '13a')
- 구두점 정규화

```python
import sacrebleu

def standard_normalize(text: str) -> str:
    """Level 2: 표준 토큰화 적용"""
    # Level 1 먼저 적용
    text = basic_cleanup(text)

    # sacrebleu 토크나이저 적용
    tokenized = sacrebleu.tokenize(text, tokenize="13a")

    return tokenized
```

**장점**:
- 학술 벤치마크(COCO, NoCaps)와 일관성
- 토큰화 차이로 인한 변동 최소화

**단점**:
- 대소문자 여전히 민감

---

### Level 3: Aggressive Normalization (강한 정규화) ⚠️
**적용**: Level 2 +
- 소문자화
- 구두점 제거

```python
def aggressive_normalize(text: str) -> str:
    """Level 3: 강한 정규화 (주의 필요)"""
    import re

    # Level 2 먼저
    text = standard_normalize(text)

    # 소문자화
    text = text.lower()

    # 구두점 제거 (선택적)
    # text = re.sub(r'[^\w\s]', ' ', text)

    return text
```

**⚠️ 주의**:
- 고유명사 정보 손실 ("New York" → "new york")
- 약어 의미 변경 ("US" → "us")
- 실제 품질 가림

**사용 사례**:
- 의미적 유사도에만 집중하는 ablation study
- 대소문자/구두점이 평가 목적이 아닌 경우

---

## 🔍 BLEU 계산 방식 선택

### 옵션 1: NLTK BLEU (현재 사용)
```python
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

smoothing = SmoothingFunction().method1  # 또는 method4
ref_tokens = [[ref.split()] for ref in refs]
pred_tokens = [pred.split() for pred in preds]
bleu = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
```

**특징**:
- 간단한 `split()` 토큰화
- Method1: 약한 스무딩 (짧은 문장에 불리)
- Method4: 강한 스무딩 (더 관대)

---

### 옵션 2: sacrebleu (표준) ⭐
```python
import sacrebleu

bleu = sacrebleu.corpus_bleu(
    preds, [refs],
    smooth_method="exp",      # 표준 스무딩
    lowercase=False,          # 대소문자 보존
    tokenize="13a",           # 표준 토크나이저
    use_effective_order=True  # 짧은 문장 안정화
)
score = bleu.score / 100.0  # 0~1 스케일
```

**특징**:
- COCO/NoCaps 벤치마크 표준
- 재현성 높음
- 논문 작성 시 신뢰도 ↑

---

## 💼 프로젝트별 권장 사항

### 1. PanoramaVLM (학습된 모델) - Level 1 + NLTK
**이유**:
- 기존 결과와의 일관성 유지
- 모델 개선 효과 명확히 측정

```python
# scripts/eval.py
def calculate_evaluation_metrics(df, output_dir, timestamp, prefix):
    # Level 1 정리만 적용
    df['prediction'] = df['prediction'].apply(basic_cleanup)
    df['reference'] = df['reference'].apply(basic_cleanup)

    # NLTK BLEU (기존 유지)
    # ... 기존 코드
```

---

### 2. HF VLM 비교 (evaluate_vlm_models.py) - Level 1 + sacrebleu
**이유**:
- 다른 연구와 비교 가능성
- 모델 간 공정한 비교

```python
# scripts/evaluate_vlm_models.py
def compute_text_metrics(predictions, references):
    # Level 1 정리
    preds = [basic_cleanup(p) for p in predictions]
    refs = [basic_cleanup(r) for r in references]

    # sacrebleu 사용
    bleu = sacrebleu.corpus_bleu(
        preds, [refs],
        smooth_method="exp",
        lowercase=False,  # 대소문자 보존
        tokenize="13a"
    )
    metrics["bleu4"] = bleu.score / 100.0
```

---

### 3. Ablation Study - 두 가지 버전 제공
**이유**:
- 원본(Level 0)으로 실제 품질 확인
- 정규화(Level 1-2)로 메트릭 안정화

```python
# 두 가지 메트릭 세트 저장
metrics_raw = compute_metrics(preds_raw, refs_raw)         # Level 0
metrics_clean = compute_metrics(preds_clean, refs_clean)   # Level 1

results = {
    "raw": metrics_raw,
    "normalized": metrics_clean
}
```

---

## 🎯 최종 권장: 단계적 적용

### Phase 1: Level 1만 모든 스크립트에 적용 ✅

**변경 사항**:
```python
# 모든 eval 스크립트에 추가
def basic_cleanup(text: str) -> str:
    """프롬프트 아티팩트 제거 (의미 보존)"""
    import re
    text = re.sub(r"<\|.*?\|>|<image>|</image>|<img>|</img>", " ", text, flags=re.I)
    text = re.sub(r"<vision_start>|<vision_end>|<image_pad>", " ", text, flags=re.I)
    text = re.sub(r"^(USER:|ASSISTANT:|Question:|Answer:)\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 메트릭 계산 전에 적용
predictions = [basic_cleanup(p) for p in raw_predictions]
references = [basic_cleanup(r) for r in raw_references]
```

**효과**:
- ✅ 프롬프트 누수 제거
- ✅ 특수 토큰 간섭 제거
- ✅ 의미 보존 (대소문자, 구두점 유지)
- ✅ 실제 품질 반영

---

### Phase 2: sacrebleu 도입 (선택적) 🔄

**조건부 적용**:
- HF VLM 비교 평가: sacrebleu 사용
- PanoramaVLM 평가: NLTK 유지 (기존 결과와 비교)

```python
# evaluate_vlm_models.py만 변경
import sacrebleu

def compute_text_metrics_hf(predictions, references):
    # Level 1 정리
    preds = [basic_cleanup(p) for p in predictions]
    refs = [basic_cleanup(r) for r in references]

    # sacrebleu
    bleu = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp", lowercase=False, tokenize="13a")
    metrics["bleu4"] = bleu.score / 100.0
    # ...
```

---

### Phase 3: 소문자화는 하지 않음 ❌

**이유**:
- 고유명사/약어 정보 손실
- 실제 품질 왜곡
- 학술 벤치마크도 대부분 case-sensitive

---

## 📊 예상 효과

### Level 0 (Raw) → Level 1 (Basic Cleanup)
- BLEU-4: **2~5%p 상승** (프롬프트 누수 제거 효과)
- 의미 보존: **100%**
- 실제 품질 반영: **높음**

### Level 1 → Level 2 (sacrebleu)
- BLEU-4: **1~3%p 상승** (토큰화 안정화)
- 벤치마크 비교: **가능**
- 실제 품질 반영: **높음**

### Level 2 → Level 3 (lowercase)
- BLEU-4: **3~8%p 상승** ⚠️ (인위적)
- 정보 손실: **중간**
- 실제 품질 반영: **낮음**

---

## 🔗 구현 우선순위

1. **즉시 적용**: `basic_cleanup()` 함수를 모든 eval 스크립트에 추가
2. **단기 적용**: HF VLM 비교에만 sacrebleu 도입
3. **장기 검토**: 멀티 레퍼런스 데이터셋 구축 (BLEU 안정화)
4. **하지 않음**: 소문자화, 구두점 제거

---

## 변경 이력

- **2025-01-XX**: 초안 작성
- **2025-01-XX**: Level 1 권장안 확정

