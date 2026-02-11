# 실험 이름 설정 - 빠른 가이드

## ✨ 한 곳에서만 설정하세요!

```yaml
# configs/default.yaml
experiment:
  name: auto                  # 또는 auto_name: true
  siglip_include_patch_res: true  # SigLIP은 "siglip2-so400m_p16_256" 처럼 세부 표기
  
training:
  # prefix는 제거됨 ❌
  # experiment.name(자동 생성)이 사용됩니다 ✅
  stages: ["vision", "resampler", "finetune"]
```

## 자동으로 생성되는 것들

### 1. 체크포인트 디렉토리
```
runs/ADDDATA_S2Q3_1_latent768_PE/vision/anyres-e2p_mlp/
runs/ADDDATA_S2Q3_1_latent768_PE/resampler/anyres-e2p_mlp/
runs/ADDDATA_S2Q3_1_latent768_PE/finetune/anyres-e2p_mlp/
```

### 2. Stage State 파일
```
runs/ADDDATA_S2Q3_1_latent768_PE_stage_state.json
```

### 3. WandB Run Name
```
SIGLIP2_QWEN3_BIMAMBA_ANYRES-E2P_PE/vision/siglip_mlp_anyres-e2p_quic360_1015-1430
```

### 4. 체크포인트 파일
```
siglip_mlp_anyres-e2p_quic360_epoch03_loss0.4523.ckpt
```

## ✅ DO

```yaml
# 1. experiment.name만 설정
experiment:
  name: "MY_EXPERIMENT_2024"

# 2. 의미 있는 이름 사용
experiment:
  name: "SIGLIP_QWEN3_ANYRES_V1"

# 3. 버전 관리
experiment:
  name: "BASELINE_V2"  # V1, V2, V3...
```

## ❌ DON'T

```yaml
# 1. training.prefix 중복 설정 (제거됨)
experiment:
  name: "MY_EXP"
training:
  prefix: "MY_EXP"  # ❌ 불필요!

# 2. 특수 문자 사용 (/, \, 공백)
experiment:
  name: "my exp/test"  # ❌ 경로 오류 발생

# 3. 너무 긴 이름
experiment:
  name: "VERY_LONG_EXPERIMENT_NAME_THAT_IS_HARD_TO_READ"  # ❌
```

## 이름 규칙 (권장)

### 패턴 1: 모델 + 버전
```yaml
experiment:
  name: "SIGLIP_QWEN_V1"
  name: "RICE_LLAMA_V2"
```

### 패턴 2: 데이터셋 + 설정
```yaml
experiment:
  name: "QUIC360_ANYRES_BASELINE"
  name: "STANFORD_E2P_LORA"
```

### 패턴 3: 날짜 + 설명
```yaml
experiment:
  name: "1017_VICREG_TUNING"
  name: "1018_FULLFT_TEST"
```

## 기존 프로젝트 마이그레이션

### 이전 (중복)
```yaml
experiment:
  name: "ADDDATA_S2Q3_1_latent768_PE"
training:
  prefix: "ADDDATA_S2Q3_1_latent768_PE"  # ❌ 제거
```

### 지금 (간단)
```yaml
experiment:
  name: "ADDDATA_S2Q3_1_latent768_PE"  # ✅ 한 곳만!
training:
  # prefix 제거됨
```

## 확인 방법

훈련 시작 시 로그를 확인하세요:

```
============================================================
📋 메타데이터에서 로드된 정보:
  - Experiment: ADDDATA_S2Q3_1_latent768_PE  👈 올바른 이름
  - Stage: vision
============================================================
```

## 더 알아보기

- [NAMING_CONVENTION.md](NAMING_CONVENTION.md) - 전체 명명 규칙
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 설정 가이드
