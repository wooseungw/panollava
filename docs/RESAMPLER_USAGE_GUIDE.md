# Resampler Configuration 사용 가이드

## 📝 개요

이제 **config.yaml에서 resampler 설정만 수정**하면 모든 것이 자동으로 처리됩니다!

## 🚀 빠른 시작

### 1. Config 파일 준비

```bash
# 템플릿 복사
cp configs/bimamba_custom.yaml configs/my_experiment.yaml

# 또는 기존 config 수정
vim configs/default.yaml
```

### 2. Resampler 설정 수정

**`configs/my_experiment.yaml`**:
```yaml
models:
  resampler_type: "bimamba"
  
  # ✨ 이 부분만 수정하면 됩니다!
  resampler_config:
    latent_dimension: 768      # LLM 입력 차원
    hidden_dim: 1024           # BiMamba hidden dimension
    depth: 4                   # Block 개수
    expand: 1.75               # Expand factor ⚠️ 중요!
    d_state: 64                # State dimension
    d_conv: 4                  # Conv kernel
    use_ln: true
    dropout: 0.05
```

### 3. Training 실행

```bash
python scripts/train.py --config configs/my_experiment.yaml
```

**자동으로 일어나는 일:**
1. ✅ Config에서 resampler 설정 읽기
2. ✅ 해당 설정으로 모델 생성
3. ✅ Training 완료 후 `checkpoint_metadata.json`에 자동 저장
4. ✅ 다음 stage에서 자동으로 이전 설정 로드

### 4. Evaluation 실행

```bash
python scripts/eval.py --checkpoint-dir runs/my_experiment/finetune/...
```

**자동으로 일어나는 일:**
1. ✅ `checkpoint_metadata.json`에서 resampler 설정 자동 로드
2. ✅ 없으면 checkpoint weights에서 자동 추론
3. ✅ 정확한 차원으로 모델 생성 → **에러 없음!**

---

## 🎯 설정 방법 (2가지)

### 방법 1: 간단하게 (개별 필드)

```yaml
models:
  resampler_type: "bimamba"
  latent_dimension: 768
  resampler_hidden_dim: 1024  # 이것만 지정해도 됨
```

- ✅ 간단함
- ⚠️ `expand`, `d_state` 등은 기본값 사용

### 방법 2: 상세하게 (권장)

```yaml
models:
  resampler_type: "bimamba"
  resampler_config:
    latent_dimension: 768
    hidden_dim: 1024
    expand: 1.75        # ✨ 명시적으로 지정!
    depth: 4
    d_state: 64
    d_conv: 4
    use_ln: true
    dropout: 0.05
    norm_first: true
```

- ✅ 모든 파라미터 명시적
- ✅ 재현성 높음
- ✅ **강력 권장!**

---

## 📊 Resampler 타입별 설정

### BiMamba (양방향 Mamba)

```yaml
models:
  resampler_type: "bimamba"
  resampler_config:
    hidden_dim: 1024       # 512, 1024, 1536, 2048
    expand: 1.75           # 1.5, 1.75, 2.0
    depth: 4               # 2, 3, 4, 6
    d_state: 64            # 16, 32, 64
    d_conv: 4              # 4 (일반적으로 고정)
```

**조합 추천:**
- **빠른 학습**: `hidden_dim: 512, expand: 2.0`
- **균형**: `hidden_dim: 1024, expand: 1.75` ⭐ (기본)
- **고성능**: `hidden_dim: 1536, expand: 1.5`

### MLP

```yaml
models:
  resampler_type: "mlp"
  resampler_config:
    hidden_dim: 1536       # 보통 latent_dim의 2배
    depth: 3               # 2, 3, 4
    use_ln: true
```

### Perceiver

```yaml
models:
  resampler_type: "perceiver"
  resampler_config:
    num_latents: 32        # Query 개수
    depth: 4               # Cross-attention layers
    heads: 8               # Attention heads
```

### QFormer

```yaml
models:
  resampler_type: "qformer"
  resampler_config:
    num_query_tokens: 64
    num_hidden_layers: 6
    num_attention_heads: 8
```

---

## 🔍 파라미터별 의미

| 파라미터 | 의미 | 영향 | 추천값 |
|---------|------|------|--------|
| `hidden_dim` | BiMamba 내부 차원 | 모델 크기, 성능 | 1024 |
| `expand` | SSM expansion factor | 파라미터 수 | 1.75 |
| `depth` | Block 개수 | 깊이, 표현력 | 4 |
| `d_state` | State space 차원 | Sequence modeling | 64 |
| `d_conv` | Conv1d kernel size | Local context | 4 |
| `dropout` | Dropout rate | 정규화 | 0.05 |

### Hidden Dim vs Expand Trade-off

```python
# 파라미터 수 계산 (대략)
params ≈ hidden_dim × expand × depth × 3

# 예시:
hidden_dim=1024, expand=1.75, depth=4
→ params ≈ 1024 × 1.75 × 4 × 3 ≈ 21.5M

hidden_dim=1536, expand=1.5, depth=4
→ params ≈ 1536 × 1.5 × 4 × 3 ≈ 27.6M
```

---

## ⚠️ 주의사항

### 1. Checkpoint 호환성

**문제 상황:**
```yaml
# Training 시
hidden_dim: 1024

# Evaluation 시 다른 값 사용
hidden_dim: 1536  # ❌ 에러!
```

**해결:**
- ✅ **그냥 아무 config 사용해도 됨!**
- ✅ Checkpoint에서 자동으로 올바른 값 로드
- ✅ Config는 참고용으로만 사용됨

### 2. 기본값 의존 금지

**나쁜 예:**
```yaml
models:
  resampler_type: "bimamba"
  # expand를 지정 안함 → 기본값(2.0) 사용
```

**좋은 예:**
```yaml
models:
  resampler_type: "bimamba"
  resampler_config:
    expand: 1.75  # ✅ 명시적으로 지정!
```

### 3. Vision Feature Dimension 확인

```yaml
models:
  vision_name: "google/siglip2-so400m-patch16-256"  # 1152-dim
  resampler_config:
    latent_dimension: 768  # ✅ LLM 입력 차원
    # input_dim은 자동으로 vision_name에서 추론 (1152)
```

---

## 📂 파일 구조

```
configs/
├── default.yaml                    # 기본 설정
├── bimamba_custom.yaml            # BiMamba 커스텀 템플릿 ⭐
├── config_resampler_example.yaml  # 예시 모음
└── my_experiment.yaml             # 내 실험 설정

runs/
└── my_experiment/
    ├── vision/
    │   └── checkpoint_metadata.json  # ✅ 설정 자동 저장
    ├── resampler/
    │   └── checkpoint_metadata.json
    └── finetune/
        └── checkpoint_metadata.json
```

---

## 🧪 테스트 예시

### 작은 모델로 빠른 실험

```yaml
models:
  vision_name: "google/siglip-base-patch16-224"  # 작은 모델
  language_model_name: "Qwen/Qwen3-0.6B"
  resampler_type: "bimamba"
  resampler_config:
    hidden_dim: 512      # 작게
    expand: 2.0          # 크게 (보상)
    depth: 3             # 얕게
```

### 대규모 모델

```yaml
models:
  vision_name: "google/siglip2-so400m-patch16-256"
  language_model_name: "Qwen/Qwen3-1.8B"
  resampler_type: "bimamba"
  resampler_config:
    hidden_dim: 2048     # 크게
    expand: 1.5          # 작게 (효율성)
    depth: 6             # 깊게
```

---

## ✅ 체크리스트

**Training 시작 전:**
- [ ] `models.resampler_config` 섹션 작성
- [ ] `hidden_dim`, `expand` 명시적으로 지정
- [ ] `experiment.name` 설정 (자동 디렉토리명)

**Training 중:**
- [ ] 첫 epoch에서 로그 확인:
  ```
  🔧 [ResamplerModule] config.resampler_hidden_dim=1024 발견
  ```
- [ ] `checkpoint_metadata.json` 생성 확인

**Evaluation 시:**
- [ ] `--checkpoint-dir`만 지정 (config 불필요)
- [ ] 로그에서 metadata 로드 확인:
  ```
  📋 Metadata에서 resampler 설정 로드
  ```

---

## 🎓 FAQ

**Q: 기존 checkpoint는 어떻게 하나요?**  
A: 그냥 사용하세요! 자동으로 weights에서 추론합니다.

**Q: Config를 바꿔도 evaluation에 영향 없나요?**  
A: 네! Checkpoint의 metadata가 최우선입니다.

**Q: Metadata 파일을 수동으로 수정해도 되나요?**  
A: 가능하지만 권장하지 않습니다. 대신 새로 training 하세요.

**Q: 다른 resampler로 바꾸려면?**  
A: Config에서 `resampler_type`만 바꾸고 새로 training.

---

## 📚 관련 문서

- **상세 기술 문서**: `docs/RESAMPLER_CONFIG_FIX.md`
- **Config 가이드**: `docs/CONFIG_GUIDE.md`
- **개선 요약**: `IMPROVEMENT_SUMMARY.md`

---

**요약**: Config만 수정하면 됩니다! 나머지는 모두 자동입니다. 🎉
