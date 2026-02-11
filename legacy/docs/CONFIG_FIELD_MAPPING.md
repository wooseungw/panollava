# Config 필드 매핑 및 검증 가이드

## 문제 상황

PanoLLaVA 코드베이스에서 config 필드 이름이 일관되지 않았습니다:

| 위치 | Vision 모델 필드명 | Language 모델 필드명 |
|------|-------------------|---------------------|
| YAML (`configs/default.yaml`) | `vision_name` | `language_model_name` |
| ModelConfig (`src/panovlm/config/schema.py`) | `vision_name` | `language_model_name` |
| train.py (일부) | `vision_model_name` ❌ | `language_model_name` ✅ |
| Legacy configs | `lm_model`, `lm_name` ❌ | - |

이로 인해:
- Config가 제대로 전달되지 않음
- 기본값으로 fallback 되어 의도하지 않은 모델 로딩
- 에러 메시지 없이 조용히 실패

## 해결 방법

### 1. 필수 필드 검증 추가

**`src/panovlm/models/model.py`**:
```python
def _validate_config(config):
    """Config 객체의 필수 속성 검증"""
    required_attrs = {
        'vision_name': "Vision encoder 모델명",
        'language_model_name': "Language 모델명",
    }
    
    missing = []
    for attr, description in required_attrs.items():
        if not hasattr(config, attr) or getattr(config, attr) is None:
            missing.append(f"  - {attr}: {description}")
    
    if missing:
        raise ValueError("Config에 다음 필수 속성이 누락되었습니다:\n" + "\n".join(missing))
```

**PanoramaVLM.__init__에서 사용**:
```python
self.config = config

# 필수 설정 검증
try:
    _validate_config(self.config)
except ValueError as e:
    raise ValueError(
        f"{str(e)}\n\n"
        f"현재 config 내용:\n"
        f"  vision_name: {getattr(self.config, 'vision_name', 'NOT SET')}\n"
        f"  language_model_name: {getattr(self.config, 'language_model_name', 'NOT SET')}"
    ) from e
```

### 2. 필드 이름 자동 매핑

**`src/panovlm/config/schema.py`**:
```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
    """
    딕셔너리에서 생성 (legacy 호환)
    
    필드 이름 자동 매핑:
    - vision_model_name → vision_name
    - lm_model → language_model_name
    - lm_name → language_model_name
    """
    config = dict(config_dict)
    
    field_aliases = {
        'vision_model_name': 'vision_name',
        'lm_model': 'language_model_name',
        'lm_name': 'language_model_name',
    }
    
    for old_name, new_name in field_aliases.items():
        if old_name in config and new_name not in config:
            config[new_name] = config.pop(old_name)
            warnings.warn(
                f"'{old_name}'은 deprecated입니다. '{new_name}'을 사용하세요.",
                DeprecationWarning
            )
    
    return cls(**config)
```

### 3. train.py 수정

**`scripts/train.py`**:
```python
def _resolve_stage_image_processing(cfg, stage_cfg):
    # 우선순위: vision_name > vision_model_name (하위 호환)
    models_cfg = cfg.get("models", {}) or {}
    if "vision_model_name" not in base:
        vision_identifier = models_cfg.get("vision_name") or models_cfg.get("vision_model_name")
        if vision_identifier:
            base["vision_model_name"] = vision_identifier
```

## 사용 방법

### 권장 방식 (표준 필드명)

```yaml
# configs/default.yaml
models:
  vision_name: "google/siglip-base-patch16-224"
  language_model_name: "Qwen/Qwen3-0.6B"
  resampler_type: "mlp"
```

```python
# Python 코드
from panovlm.config import ModelConfig

config = ModelConfig(
    vision_name='google/siglip-base-patch16-224',
    language_model_name='Qwen/Qwen3-0.6B',
    resampler_type='mlp'
)
```

### Legacy 호환 (자동 변환됨)

```python
# 이전 코드 (자동으로 매핑됨)
config = ModelConfig.from_dict({
    'vision_model_name': 'google/siglip-base-patch16-224',  # → vision_name
    'lm_model': 'Qwen/Qwen3-0.6B',  # → language_model_name
})

# DeprecationWarning 표시:
# 'vision_model_name'은 deprecated입니다. 'vision_name'을 사용하세요.
# 'lm_model'은 deprecated입니다. 'language_model_name'을 사용하세요.
```

## 테스트

### Config 검증 테스트

```bash
python scripts/test_config.py
```

**테스트 내용**:
1. ✅ 표준 필드명 사용
2. ✅ Legacy 필드명 자동 매핑
3. ✅ 필수 필드 누락 감지
4. ✅ YAML config 로딩
5. ✅ 모델 인스턴스화 (선택적)

### 예상 출력

```
==========================================================
Config 필드 매핑 테스트
==========================================================

1. 표준 필드 이름 사용:
   ✓ vision_name: google/siglip-base-patch16-224
   ✓ language_model_name: Qwen/Qwen3-0.6B

2. Legacy 필드 이름 (vision_model_name):
   ✓ vision_name: google/siglip-base-patch16-224
   ✓ Automatically mapped vision_model_name → vision_name

3. Legacy 필드 이름 (lm_model):
   ✓ language_model_name: Qwen/Qwen3-0.6B
   ✓ Automatically mapped lm_model → language_model_name

5. 필수 필드 검증:
   ✓ 검증 성공 - 필수 필드 누락 감지

==========================================================
모든 테스트 통과!
==========================================================
```

## 에러 메시지

### 필수 필드 누락 시

```python
ValueError: Config에 다음 필수 속성이 누락되었습니다:
  - vision_name: Vision encoder 모델명 (예: 'google/siglip-base-patch16-224')

현재 config 내용:
  vision_name: NOT SET
  language_model_name: Qwen/Qwen3-0.6B

해결 방법:
  1. YAML 설정 파일에 해당 필드 추가
  2. 또는 ModelConfig 생성 시 직접 전달:
     config = ModelConfig(
         vision_name='google/siglip-base-patch16-224',
         language_model_name='Qwen/Qwen2.5-0.5B-Instruct'
     )
```

### 필드명 deprecation 경고

```python
DeprecationWarning: 'vision_model_name'은 deprecated입니다. 'vision_name'을 사용하세요.
```

## 마이그레이션 가이드

### YAML 설정 파일

**Before**:
```yaml
models:
  vision_model_name: "google/siglip-base-patch16-224"  # ❌ deprecated
  lm_model: "Qwen/Qwen3-0.6B"  # ❌ deprecated
```

**After**:
```yaml
models:
  vision_name: "google/siglip-base-patch16-224"  # ✅ 권장
  language_model_name: "Qwen/Qwen3-0.6B"  # ✅ 권장
```

### Python 코드

**Before**:
```python
config = {
    'vision_model_name': 'google/siglip-base-patch16-224',
    'lm_model': 'Qwen/Qwen3-0.6B',
}
```

**After**:
```python
config = {
    'vision_name': 'google/siglip-base-patch16-224',
    'language_model_name': 'Qwen/Qwen3-0.6B',
}
```

## 영향받는 파일

### 수정된 파일
- ✅ `src/panovlm/models/model.py`: 필수 필드 검증 추가
- ✅ `src/panovlm/config/schema.py`: 필드 이름 자동 매핑
- ✅ `scripts/train.py`: vision_name/vision_model_name 모두 지원

### 테스트 파일
- ✅ `scripts/test_config.py`: Config 검증 테스트

### 문서
- ✅ `docs/CONFIG_FIELD_MAPPING.md`: 이 문서

## 요약

| 변경 사항 | 효과 |
|----------|------|
| 필수 필드 검증 | 누락 시 명확한 에러 메시지 |
| 필드 이름 자동 매핑 | Legacy 코드 호환성 유지 |
| Deprecation 경고 | 점진적 마이그레이션 유도 |
| 테스트 스크립트 | Config 문제 사전 감지 |

✅ **결과**: Config가 제대로 전달되지 않는 문제 해결
✅ **호환성**: 기존 코드 동작 보장
✅ **유지보수성**: 표준 필드명으로 점진적 이전
