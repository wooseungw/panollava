# 토크나이저 패딩 방향 설정 가이드

## 개요

PanoLLaVA는 이제 **LLM 모델의 원래 토크나이저 설정을 존중**하여 패딩 방향을 자동으로 구성합니다.

## 변경 사항

### Before (하드코딩)
```python
# 항상 "right" padding으로 고정
self.tokenizer.padding_side = "right"
```

**문제점**:
- Llama, Qwen 등 decoder-only 모델은 보통 **left padding** 사용
- 원래 설정 무시로 인한 잠재적 성능 저하
- 모델 타입에 관계없이 일괄 적용

### After (자동 감지)
```python
# 1. 원래 토크나이저 설정 저장
self._original_padding_side = self.tokenizer.padding_side

# 2. 원래 설정 우선 사용
if original_side:
    self.tokenizer.padding_side = original_side
    print(f"Padding side: '{original_side}' (from original tokenizer)")
```

## 패딩 방향 결정 로직

우선순위 순서:

1. **원래 토크나이저 설정** (최우선)
   - `AutoTokenizer.from_pretrained()`로 로딩한 원본 설정 사용
   
2. **모델 타입별 권장 설정** (fallback)
   - Llama, Mistral, Qwen → `left` padding
   - T5, BART → `right` padding
   
3. **기본값** (최후 수단)
   - `right` padding

## 모델별 패딩 방향

| 모델 | 기본 Padding | 이유 |
|------|-------------|------|
| Llama | `left` | Decoder-only, 생성 시 왼쪽 패딩 필요 |
| Qwen | `left` | Decoder-only, 왼쪽 패딩 권장 |
| Mistral | `left` | Decoder-only 아키텍처 |
| Gemma | `left` | Decoder-only 아키텍처 |
| T5 | `right` | Encoder-decoder, 오른쪽 패딩 |
| BART | `right` | Encoder-decoder 아키텍처 |

## 로그 메시지

### 원래 설정 사용
```
[Tokenizer Setup] Padding side: 'left' (from original tokenizer config)
```

### 권장 설정 사용
```
[Tokenizer Setup] Padding side: 'left' (recommended for Qwen/Qwen3-0.6B)
```

### 기본값 사용
```
[Tokenizer Setup] Padding side: 'right' (default)
```

## 예시

### Qwen 모델
```python
from panovlm.models.model import PanoramaVLM
from panovlm.config import ModelConfig

config = ModelConfig(
    vision_name='google/siglip-base-patch16-224',
    language_model_name='Qwen/Qwen3-0.6B',  # Left padding
)

model = PanoramaVLM(config=config)
# [Tokenizer Setup] Padding side: 'left' (from original tokenizer config)

print(model.tokenizer.padding_side)  # 'left'
```

### Llama 모델
```python
config = ModelConfig(
    vision_name='google/siglip-base-patch16-224',
    language_model_name='meta-llama/Llama-3.2-1B',  # Left padding
)

model = PanoramaVLM(config=config)
# [Tokenizer Setup] Padding side: 'left' (from original tokenizer config)

print(model.tokenizer.padding_side)  # 'left'
```

### T5 모델 (Encoder-Decoder)
```python
config = ModelConfig(
    vision_name='google/siglip-base-patch16-224',
    language_model_name='google/flan-t5-base',  # Right padding
)

model = PanoramaVLM(config=config)
# [Tokenizer Setup] Padding side: 'right' (from original tokenizer config)

print(model.tokenizer.padding_side)  # 'right'
```

## 수동 오버라이드

필요한 경우 수동으로 패딩 방향을 변경할 수 있습니다:

```python
# 모델 생성 후
model = PanoramaVLM(config=config)

# 패딩 방향 수동 변경 (권장하지 않음)
model.tokenizer.padding_side = 'right'
print(f"Manual override: {model.tokenizer.padding_side}")
```

⚠️ **주의**: 수동 변경은 모델의 원래 설계와 맞지 않을 수 있으므로 권장하지 않습니다.

## Left Padding vs Right Padding

### Left Padding (Decoder-only 모델 권장)

```
Input:  "Describe the image"
Tokens: [PAD] [PAD] [PAD] Describe the image

장점:
✅ 생성 시 마지막 토큰이 중요 (attention mask 활용)
✅ Autoregressive generation에 최적
✅ Llama, Qwen 등의 기본 설정

단점:
⚠️ Position encoding이 올바르게 동작해야 함
```

### Right Padding (Encoder-decoder 권장)

```
Input:  "Describe the image"
Tokens: Describe the image [PAD] [PAD] [PAD]

장점:
✅ Encoder 입력에 자연스러움
✅ 위치 정보가 순차적
✅ T5, BART의 기본 설정

단점:
⚠️ Decoder-only 생성에는 비효율적
```

## 디버깅

### 현재 패딩 설정 확인
```python
print(f"Padding side: {model.tokenizer.padding_side}")
print(f"Pad token: {model.tokenizer.pad_token}")
print(f"Pad token ID: {model.tokenizer.pad_token_id}")
```

### 패딩 동작 테스트
```python
from transformers import AutoTokenizer

# 원본 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(f"Original padding_side: {tokenizer.padding_side}")

# 패딩 테스트
texts = ["Short", "This is a much longer sentence"]
encoded = tokenizer(texts, padding=True, return_tensors='pt')
print(encoded['input_ids'])
```

## 관련 파일

- **모델 코드**: `src/panovlm/models/model.py`
  - `__init__`: 원래 패딩 설정 저장
  - `_setup_tokenizer()`: 패딩 방향 구성

## 참고 자료

- [HuggingFace Tokenizer Padding](https://huggingface.co/docs/transformers/pad_truncation)
- [Left vs Right Padding in Language Models](https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-use-left-padding/57654)
- [Qwen2 Tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json)

## 요약

✅ **자동 감지**: LLM 원래 설정을 자동으로 감지하여 적용  
✅ **안전한 Fallback**: 원래 설정이 없으면 모델 타입별 권장값 사용  
✅ **명확한 로그**: 어떤 설정이 적용되었는지 명확히 표시  
✅ **하위 호환**: 기존 코드 동작 보장  

이제 각 LLM 모델의 특성에 맞는 최적의 패딩 방향이 자동으로 설정됩니다! 🎯
