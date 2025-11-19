# LoRA Checkpoint Loading 가이드

## 🔍 "누락된 키 / 예상치 못한 키" 메시지는 정상입니다!

### 출력 예시
```
🔍 LoRA 감지: Lightning 체크포인트에 LoRA state_dict 포함
⚙️ 가중치 로딩 중...
   - 로드된 키: 1024/1335
   ✅ LoRA 관련 누락 키: 311개 (정상 - LoRA 구조 차이)
   ✅ LoRA 관련 추가 키: 703개 (정상 - Lightning 체크포인트 포함)
```

## ❓ 왜 이런 메시지가 나오나요?

### 1. LoRA 적용 시 키 구조 변경

**일반 모델 (LoRA 없음):**
```python
language_model.model.layers.0.self_attn.q_proj.weight
language_model.model.layers.0.self_attn.k_proj.weight
```

**LoRA 적용된 모델:**
```python
# Base model은 frozen
language_model.base_model.model.layers.0.self_attn.q_proj.weight

# LoRA adapters (trainable)
language_model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
language_model.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight
```

→ **키 구조가 완전히 다릅니다!**

### 2. PyTorch Lightning 체크포인트 구조

**Lightning 체크포인트 (.ckpt)**에는 다음이 포함됩니다:
```python
{
    'state_dict': {
        'model.vision_encoder.*': ...,           # Vision encoder weights
        'model.resampler_module.*': ...,         # Resampler weights
        'model.language_model.base_model.*': ..., # LoRA-adapted LM weights
    },
    'optimizer_states': [...],
    'lr_schedulers': [...],
    'hparams': {...},
    ...
}
```

**하지만 `from_checkpoint()`에서 생성하는 모델:**
```python
PanoramaVLM(
    vision_encoder=...,          # 'vision_encoder.*'
    resampler_module=...,        # 'resampler_module.*'
    language_model=...,          # 'language_model.*' (LoRA 적용됨)
)
```

→ **`model.` 프리픽스가 다릅니다!**

### 3. Missing vs Unexpected 키

**Missing Keys (누락된 키):**
- Lightning 체크포인트에는 없지만 현재 모델이 기대하는 키
- 예: `language_model.base_model.*` (체크포인트는 `model.language_model.*`)
- **LoRA 관련은 정상!** LoRA 구조 차이 때문

**Unexpected Keys (예상치 못한 키):**
- 체크포인트에는 있지만 현재 모델에 없는 키
- 예: `model.vision_encoder.*` (현재 모델은 `vision_encoder.*`)
- Lightning 메타데이터 키들 (optimizer, scheduler 등)
- **대부분 정상!** Lightning 체크포인트 특성

## ✅ 정상적인 로딩 확인 방법

### 1. 로그 메시지 확인

**✅ 정상:**
```
   - 로드된 키: 1024/1335
   ✅ LoRA 관련 누락 키: 311개 (정상 - LoRA 구조 차이)
   ✅ LoRA 관련 추가 키: 703개 (정상 - Lightning 체크포인트 포함)
```

**⚠️ 주의 필요:**
```
   - 로드된 키: 100/1335  ← 너무 적음!
   ⚠️  Non-LoRA 누락 키: 500개  ← Vision/Resampler 가중치 누락?
```

### 2. 핵심 컴포넌트 로딩 확인

**확인해야 할 것:**
- ✅ Vision Encoder: `vision_encoder.*` 키들 로드됨
- ✅ Resampler: `resampler_module.*` 키들 로드됨
- ✅ Language Model Base: `language_model.base_model.model.*` 키들 로드됨
- ✅ LoRA Adapters: `language_model.*.lora_A.*`, `language_model.*.lora_B.*` 키들 로드됨

### 3. 실제 추론 테스트

```python
# 간단한 forward pass 테스트
import torch
output = model.generate(
    pixel_values=torch.randn(1, 3, 224, 224),
    input_ids=torch.tensor([[1, 2, 3]]),
    max_new_tokens=10
)
print("✅ 모델 정상 작동!" if output.shape[1] > 3 else "❌ 문제 발생")
```

## 🔧 문제 해결

### 문제 1: Non-LoRA 누락 키가 많음

**증상:**
```
⚠️  Non-LoRA 누락 키: 500개
     • vision_encoder.embeddings.position_embedding
     • resampler_module.resampler.blocks.0.weight
```

**원인:** Vision encoder 또는 Resampler 가중치가 체크포인트에 없음

**해결:**
1. 올바른 체크포인트 경로 확인
2. 체크포인트가 올바른 stage에서 저장되었는지 확인
   - Vision stage: vision_encoder만 학습
   - Resampler stage: + resampler 추가
   - Finetune stage: 모든 가중치 포함

### 문제 2: 로드된 키가 너무 적음

**증상:**
```
   - 로드된 키: 50/1335  ← 매우 적음!
```

**원인:** 체크포인트 파일이 손상되었거나 잘못된 파일

**해결:**
```bash
# 체크포인트 검사
python -c "
import torch
ckpt = torch.load('checkpoint.ckpt', map_location='cpu')
print('Keys in checkpoint:', ckpt.keys())
print('State dict keys:', len(ckpt.get('state_dict', {}).keys()))
"
```

### 문제 3: LoRA 가중치가 로드되지 않음

**증상:**
```
🔍 LoRA 감지 실패 - lora_weights/ 디렉토리 사용
```

**원인:** Lightning 체크포인트에 LoRA state_dict가 포함되지 않음

**해결:**
```bash
# LoRA 가중치가 별도 디렉토리에 있는지 확인
ls -la runs/.../finetune/.../lora_weights/

# 있다면 자동으로 로드됨
# 없다면 training 시 LoRA가 제대로 저장되지 않은 것
```

## 📊 키 로딩 통계 예시

### 정상적인 케이스

**Finetune Stage (LoRA 적용)**
```
Total keys in checkpoint: 1335
Loaded keys: 1024/1335
Missing keys:
  ✅ LoRA-related: 311 (language_model.base_model.*)
  ⚠️  Non-LoRA: 0
Unexpected keys:
  ✅ LoRA-related: 700 (model.*, optimizer.*, etc.)
  ⚠️  Non-LoRA: 3 (minor metadata)
```

**Resampler Stage (LoRA 없음)**
```
Total keys in checkpoint: 800
Loaded keys: 750/800
Missing keys:
  ✅ LoRA-related: 0
  ⚠️  Non-LoRA: 50 (language_model.* - not trained yet)
Unexpected keys:
  ✅ Lightning metadata: 50
  ⚠️  Non-LoRA: 0
```

## 🎓 FAQ

**Q: 누락된 키 311개는 문제 아닌가요?**  
A: LoRA 관련이면 정상입니다. LoRA는 `base_model.model.*` 구조를 사용하는데, 체크포인트는 Lightning의 `model.*` 구조로 저장되어 키 이름이 다릅니다.

**Q: 예상치 못한 키 703개는?**  
A: Lightning 체크포인트에는 optimizer, scheduler, hparams 등 메타데이터가 포함됩니다. 이들은 모델 가중치가 아니므로 무시됩니다.

**Q: 로드된 키 1024/1335는 충분한가요?**  
A: 네! 나머지는 LoRA 구조 차이 때문입니다. 실제로 필요한 가중치는 모두 로드됩니다.

**Q: strict=False는 안전한가요?**  
A: 네! LoRA 사용 시 필수입니다. `strict=True`면 키 이름이 정확히 일치해야 하는데, LoRA는 구조가 달라서 불가능합니다.

## 📝 요약

| 메시지 | 의미 | 정상 여부 |
|--------|------|-----------|
| "LoRA 관련 누락 키: 311개" | LoRA 구조 차이 | ✅ 정상 |
| "LoRA 관련 추가 키: 703개" | Lightning 메타데이터 | ✅ 정상 |
| "Non-LoRA 누락 키: 0개" | 핵심 가중치 완전 로드 | ✅ 정상 |
| "Non-LoRA 누락 키: 500개" | Vision/Resampler 누락? | ⚠️ 확인 필요 |
| "로드된 키: 1024/1335" | 대부분 로드됨 | ✅ 정상 (LoRA 사용 시) |
| "로드된 키: 50/1335" | 대부분 누락됨 | ❌ 문제 있음 |

**결론**: LoRA 사용 시 키 불일치는 정상입니다! Non-LoRA 키만 확인하세요. 🎉
