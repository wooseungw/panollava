# LoRA 키 "누락" 로그 설명 - 정상 작동 중입니다!

## ✅ 결론: 정상 작동

**"누락된 키 311개" 로그는 정상적인 현상입니다!**

### 요약
- 체크포인트 로딩 시 311개 누락 키가 보이지만 **문제 아님**
- LoRA는 `lora_weights` 폴더에서 별도로 로드됨 ✓
- 최종적으로 LoRA가 올바르게 적용됨 ✓

---

## 왜 "누락된 키" 로그가 나타나는가?

### 1. 학습 시 저장되는 구조 (PeftModel)

```python
# 학습 중 PanoramaVLM은 PeftModel을 포함
state_dict = {
    'model.language_model.base_model.model.model.layers.0.*.lora_A.default.weight',
    'model.language_model.base_model.model.model.layers.0.*.lora_B.default.weight',
    'model.language_model.base_model.model.lm_head.weight',
    ...
}
```

**프리픽스**: `model.language_model.base_model.model.*` (PeftModel 구조)

### 2. 로딩 시 기대하는 구조 (일반 모델)

```python
# 새로 초기화된 PanoramaVLM은 일반 모델
expected_keys = {
    'language_model.model.layers.0.self_attn.q_proj.weight',
    'language_model.model.layers.0.self_attn.k_proj.weight',
    'language_model.model.embed_tokens.weight',
    ...
}
```

**프리픽스**: `language_model.model.*` (일반 모델 구조)

### 3. 프리픽스 불일치 → 누락 키 발생

```
체크포인트 키: model.language_model.base_model.model.model.layers.X.*
모델 기대 키:                language_model.model.layers.X.*
                           ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                           불일치! → 누락으로 분류
```

**로그 결과**:
```
⚙️  가중치 로딩 중...
   - 로드된 키: 944 (vision, resampler, projector 등)
   - 누락된 키: 311 (language_model의 LoRA 키)
   - 예상치 못한 키: 703 (체크포인트의 base_model.* 프리픽스 키)
```

### 4. 그래서 별도로 LoRA 폴더에서 로드

```python
# model.py:1098-1110
🔍 LoRA 가중치 자동 감지: runs/.../lora_weights
🔧 LoRA 가중치 로딩: runs/.../lora_weights

# PeftModel.from_pretrained() 실행
Converting to PeftModel.from_pretrained...
✓ LoRA weights loaded via PeftModel
   ✅ LoRA 로딩 성공 - Rank: 32, Alpha: 64
```

---

## 실제 작동 확인

### 체크포인트 로딩 과정

```
1단계: 체크포인트 로드 (.ckpt 파일)
├─ Vision encoder ✓
├─ Resampler ✓
├─ Projector ✓
└─ Language model → 311개 누락 (프리픽스 불일치, 정상)

2단계: LoRA 자동 감지
└─ lora_weights 폴더 발견 ✓

3단계: LoRA 별도 로드 (PeftModel)
└─ ✅ LoRA 적용 성공 (r=32, alpha=64)
```

### LoRA 정보 확인

```python
lora_info = model.get_lora_info()
# {
#   'is_lora_enabled': True,      ✓
#   'lora_r': 32,                  ✓
#   'lora_alpha': 64,              ✓
#   'target_modules': {...}        ✓
# }
```

---

## 파일 구조

### 체크포인트 파일 (.ckpt)
```
총 1255개 키:
├─ vision/resampler/projector: 552개
│  └─ ✓ 로드 성공
│
└─ language_model: 703개
   ├─ model.language_model.base_model.model.lm_head.weight
   ├─ model.language_model.base_model.model.model.layers.X.*.lora_A.default.weight
   └─ model.language_model.base_model.model.model.layers.X.*.lora_B.default.weight
      └─ ✗ 프리픽스 불일치로 누락 (정상 - 별도 로드됨)
```

### lora_weights 폴더 (adapter_model.bin)
```
총 394개 키:
├─ LoRA adapter: 392개
│  ├─ base_model.model.model.layers.X.*.lora_A.weight
│  └─ base_model.model.model.layers.X.*.lora_B.weight
│     └─ ✓ PeftModel.from_pretrained()로 로드
│
└─ modules_to_save: 2개
   ├─ base_model.model.model.embed_tokens.weight
   └─ base_model.model.lm_head.weight
      └─ ✓ 함께 로드
```

---

## 코드 흐름 (정상)

```
eval.py:1153
└─> load_model_and_lora()
    └─> PanoramaVLM.from_checkpoint()
        ├─ 1단계: 체크포인트 로드
        │   └─ load_state_dict(strict=False)
        │       ├─ 성공: vision, resampler, projector (944개)
        │       └─ 누락: language_model (311개) ← 프리픽스 불일치 (정상)
        │
        ├─ 2단계: LoRA 자동 감지
        │   └─ lora_weights 폴더 발견
        │
        └─ 3단계: LoRA 로드
            └─ PeftModel.from_pretrained()
                └─ ✅ LoRA 적용 성공 (r=32, alpha=64)
```

**최종 결과**:
- Vision encoder: 체크포인트에서 로드 ✓
- Resampler: 체크포인트에서 로드 ✓
- Projector: 체크포인트에서 로드 ✓
- Language model: lora_weights에서 로드 ✓

---

## 진단 방법

### 1. LoRA 로딩 확인
```bash
python scripts/eval.py --config configs/default.yaml --csv-input data/test.csv 2>&1 | grep "LoRA"
```

**성공 시 (정상):**
```
🔍 LoRA 가중치 자동 감지: runs/.../lora_weights
🔧 LoRA 가중치 로딩: runs/.../lora_weights
✓ LoRA weights loaded via PeftModel
   ✅ LoRA 로딩 성공 - Rank: 32, Alpha: 64
```

**실패 시 (문제):**
```
Warning: PEFT not available. Cannot load LoRA weights.
❌ LoRA 로딩 실패
```

### 2. LoRA 정보 확인 (Python)
```python
from src.panovlm.models.model import PanoramaVLM

model = PanoramaVLM.from_checkpoint(
    'runs/.../checkpoint.ckpt',
    device='cuda'
)

lora_info = model.get_lora_info()
print(lora_info)
# {'is_lora_enabled': True, 'lora_r': 32, 'lora_alpha': 64, ...}
```

### 3. PEFT 설치 확인
```bash
source /data/3_lib/miniconda3/bin/activate pano
python -c "import peft; print(f'PEFT {peft.__version__} installed')"
# PEFT 0.17.1 installed
```

---

## FAQ

### Q1: "누락된 키 311개"가 정상인가요?
**A**: 네, 정상입니다. 체크포인트의 LoRA 키는 프리픽스가 다르기 때문에 누락되지만,
별도의 `lora_weights` 폴더에서 올바르게 로드됩니다.

### Q2: LoRA 없이 평가하려면?
**A**: `lora_weights` 폴더를 제거하거나 다른 곳으로 이동하면 됩니다.
```bash
mv runs/.../lora_weights runs/.../lora_weights.bak
```

### Q3: 체크포인트에 LoRA가 포함되는 이유는?
**A**: PyTorch Lightning이 전체 모델을 저장할 때 PeftModel의 구조까지 포함되기 때문입니다.
하지만 로딩 시에는 프리픽스 불일치로 사용되지 않습니다.

### Q4: lora_weights 폴더가 없으면?
**A**: 자동 감지가 실패하고 LoRA 없이 평가됩니다.
로그에서 `⚠️  LoRA 경로가 존재하지 않습니다` 메시지를 확인할 수 있습니다.

### Q5: 덮어씌울 때 문제가 되나요?
**A**: 아니요, 문제없습니다. 체크포인트의 LoRA 키는 프리픽스 불일치로 무시되고,
`lora_weights` 폴더의 LoRA가 `PeftModel.from_pretrained()`를 통해 올바르게 적용됩니다.
두 소스가 충돌하지 않습니다.

---

## 요약 표

| 로그 메시지 | 의미 | 정상? |
|------------|------|-------|
| "누락된 키: 311" | 체크포인트의 LoRA 키 프리픽스 불일치 | ✅ 정상 |
| "예상치 못한 키: 703" | 체크포인트의 base_model 프리픽스 | ✅ 정상 |
| "🔍 LoRA 가중치 자동 감지" | lora_weights 폴더 발견 | ✅ 정상 |
| "✅ LoRA 로딩 성공" | LoRA가 올바르게 적용됨 | ✅ 정상 |
| "❌ LoRA 로딩 실패" | PEFT 없거나 파일 오류 | ❌ 문제 |

---

## 결론

### 핵심 포인트

1. **"누락된 키" 로그는 무시해도 됩니다**
   - 프리픽스 불일치로 인한 정상적인 현상

2. **"LoRA 로딩 성공" 메시지가 나오면 정상 작동 중입니다**
   - lora_weights에서 올바르게 로드됨

3. **평가 결과는 학습된 LoRA가 적용된 상태입니다**
   - 체크포인트 LoRA와 lora_weights LoRA가 충돌하지 않음

4. **덮어씌우기 문제 없습니다**
   - 체크포인트: 프리픽스 불일치로 무시
   - lora_weights: PeftModel로 올바르게 적용
   - 두 소스가 독립적으로 처리됨

### 확인 방법

평가 실행 시 다음 로그가 나오면 정상입니다:
```
✅ LoRA 로딩 성공 - Rank: 32, Alpha: 64
```

이 메시지가 보이면 LoRA가 올바르게 적용된 상태로 평가가 진행됩니다!
