# CoT(Chain-of-Thought) 토큰 제거 가이드

## 개요
`eval.py`의 `basic_cleanup()` 함수가 CoT 토큰과 그 안의 내용을 자동으로 제거합니다.

## 제거되는 CoT 토큰 목록

### 1. Think 계열
```xml
<think>이것은 내부 사고 과정입니다</think>
<thinking>분석 중...</thinking>
<thought>생각 중...</thought>
```

### 2. Reasoning 계열
```xml
<reasoning>추론 과정입니다</reasoning>
<reason>이유는...</reason>
```

### 3. Analysis 계열
```xml
<analysis>분석 내용</analysis>
<reflection>성찰 내용</reflection>
```

## 동작 방식

### Before (원본 텍스트)
```
<think>이 이미지에는 침대와 소파가 보입니다. 방의 레이아웃을 분석해야 합니다.</think>
이 방은 침실로 보이며 침대, 소파, 테이블이 있습니다.
```

### After (정제된 텍스트)
```
이 방은 침실로 보이며 침대, 소파, 테이블이 있습니다.
```

## 처리되는 패턴

### 1. 완전한 쌍 (내용 포함 완전 제거)
```python
# DOTALL 플래그로 줄바꿈 포함 모두 제거
<think>
여러 줄에 걸친
내부 사고 내용
</think>
→ 완전히 제거됨
```

### 2. 대소문자 무관
```python
<THINK>...</THINK>   → 제거
<Think>...</Think>   → 제거
<think>...</think>   → 제거
```

### 3. 닫히지 않은 태그도 제거
```python
<think>        → 제거
</think>       → 제거
<thinking>     → 제거
```

## 사용 예시

### Python 코드
```python
from eval import basic_cleanup

# 예시 1: 단순 CoT
text1 = "<think>Let me analyze this image.</think> The room has a bed."
result1 = basic_cleanup(text1)
print(result1)  # "The room has a bed."

# 예시 2: 여러 CoT 토큰
text2 = """
<think>First, I see a bed</think>
The bedroom has <reasoning>because there's furniture</reasoning> a cozy atmosphere.
<analysis>The color scheme is warm</analysis>
"""
result2 = basic_cleanup(text2)
print(result2)  # "The bedroom has a cozy atmosphere."

# 예시 3: 중첩된 내용
text3 = "<think>This is <important>nested</important> content</think> Final answer."
result3 = basic_cleanup(text3)
print(result3)  # "Final answer."
```

### 명령줄 테스트
```bash
# eval.py 실행 시 자동으로 적용됨
python scripts/eval.py \
    --config runs/.../config.yaml \
    --csv-input data/test.csv
```

## 기술적 세부사항

### 정규식 플래그
- `re.I`: 대소문자 무관 (case-insensitive)
- `re.DOTALL`: `.`이 줄바꿈 포함 모든 문자 매칭

### 처리 순서
1. **CoT 태그 제거** (최우선)
2. 일반 특수 토큰 제거 (`<image>`, `<|im_start|>` 등)
3. 역할 태그 제거 (`USER:`, `ASSISTANT:`)
4. 공백 정리
5. 프롬프트 누수 제거 (`Think...`, `Let me think...`)

### 제거 전략
```python
# 탐욕적 매칭 방지: .*? 사용
text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.DOTALL)
#                       ^^^ 최소 매칭 (non-greedy)

# 잘못된 예 (탐욕적 매칭):
# text = re.sub(r"<think>.*</think>", "", text)
# 결과: <think>A</think> text <think>B</think> → 모두 제거됨
```

## 추가 커스터마이징

### 새로운 CoT 토큰 추가
`eval.py`의 `basic_cleanup()` 함수에 패턴 추가:

```python
# 새로운 토큰 추가 예시
text = re.sub(r"<internal>.*?</internal>", "", text, flags=re.I | re.DOTALL)
text = re.sub(r"<draft>.*?</draft>", "", text, flags=re.I | re.DOTALL)
```

### 특정 토큰만 보존하고 싶을 때
```python
# <summary>는 보존하고 싶은 경우
# basic_cleanup()에서 해당 패턴 제외하거나
# 별도 함수 작성

def cleanup_except_summary(text: str) -> str:
    # 1. <summary> 임시 보호
    summaries = re.findall(r"<summary>.*?</summary>", text, flags=re.I | re.DOTALL)
    placeholders = []
    for i, summary in enumerate(summaries):
        placeholder = f"___SUMMARY_{i}___"
        text = text.replace(summary, placeholder)
        placeholders.append((placeholder, summary))
    
    # 2. 기본 정제
    text = basic_cleanup(text)
    
    # 3. <summary> 복원
    for placeholder, summary in placeholders:
        text = text.replace(placeholder, summary)
    
    return text
```

## 테스트 케이스

### 성공 케이스
```python
# ✅ 단순 제거
assert basic_cleanup("<think>test</think>A") == "A"

# ✅ 여러 줄
assert basic_cleanup("<think>\nline1\nline2\n</think>A") == "A"

# ✅ 여러 태그
assert basic_cleanup("<think>A</think>B<reasoning>C</reasoning>D") == "B D"

# ✅ 대소문자
assert basic_cleanup("<THINK>test</THINK>A") == "A"
```

### 엣지 케이스
```python
# ⚠️ 닫히지 않은 태그
text = "<think>A"
result = basic_cleanup(text)
# 결과: "A" (시작 태그만 제거)

# ⚠️ 중첩된 태그
text = "<think><think>nested</think></think>A"
result = basic_cleanup(text)
# 결과: "A" (내부 중첩 포함 제거)

# ⚠️ 태그 이름 부분 매칭
text = "<thinking-hard>test</thinking-hard>A"
result = basic_cleanup(text)
# 결과: "test A" (정확한 태그 이름만 매칭됨)
```

## 성능 고려사항

### 정규식 컴파일 (대량 처리 시)
```python
import re

# 반복 사용 시 성능 향상
COT_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.I | re.DOTALL),
    re.compile(r"<thinking>.*?</thinking>", re.I | re.DOTALL),
    re.compile(r"<reasoning>.*?</reasoning>", re.I | re.DOTALL),
]

def fast_cleanup(text: str) -> str:
    for pattern in COT_PATTERNS:
        text = pattern.sub("", text)
    return text
```

### 배치 처리
```python
# DataFrame 전체에 적용
import pandas as pd

df['cleaned_prediction'] = df['prediction'].apply(basic_cleanup)
df['cleaned_reference'] = df['reference'].apply(basic_cleanup)
```

## 문제 해결

### 1. CoT 내용이 제거되지 않을 때
```python
# 원인: 태그 이름 오타 또는 변형
# 해결: 로그 확인 및 패턴 추가

import logging
logger.debug(f"Before cleanup: {text[:100]}")
cleaned = basic_cleanup(text)
logger.debug(f"After cleanup: {cleaned[:100]}")
```

### 2. 필요한 내용이 제거될 때
```python
# 원인: "<think>"가 실제 답변에 포함된 경우
# 해결: 더 엄격한 패턴 사용

# 예: "I think" 보존하고 "<think>" 태그만 제거
# (현재 구현은 태그만 제거하므로 문제 없음)
```

### 3. 성능 문제
```python
# 원인: re.DOTALL로 인한 백트래킹
# 해결: 데이터 크기 제한 또는 타임아웃 설정

import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Cleanup timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5초 타임아웃
try:
    cleaned = basic_cleanup(very_long_text)
finally:
    signal.alarm(0)
```

## 관련 문서
- **EVALUATION_PREPROCESSING_GUIDE.md**: 전체 전처리 파이프라인
- **EVALUATION_METRICS_GUIDE.md**: 메트릭 계산 방식
- **eval.py**: 실제 구현 코드

## 변경 이력
- **2025-10-23**: CoT 토큰 제거 기능 추가
  - `<think>`, `<reasoning>`, `<analysis>` 계열 토큰 지원
  - DOTALL 플래그로 여러 줄 처리
  - 닫히지 않은 태그 처리
