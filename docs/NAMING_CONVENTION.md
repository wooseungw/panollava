# PanoLLaVA Naming Convention Guide

## 개요

PanoLLaVA는 **단일 소스 (experiment.name)**에서 모든 이름을 자동 생성하여 일관성을 보장합니다.

## ✨ 핵심 원칙: Single Source of Truth

```yaml
# configs/default.yaml
experiment:
  name: "ADDDATA_S2Q3_1_latent768_PE"  # 👈 여기에만 이름을 설정!
  
training:
  # prefix는 제거됨 - experiment.name을 자동 사용
  stages: ["vision", "resampler", "finetune"]
```

**더 이상 중복 설정 불필요!** ❌ `training.prefix` 제거됨

## 자동 생성되는 이름들

### 1. 체크포인트 디렉토리
```
runs/{experiment_name}/{stage}/{crop_strategy}_{resampler}/

예시:
runs/ADDDATA_S2Q3_1_latent768_PE/vision/anyres-e2p_mlp/
runs/ADDDATA_S2Q3_1_latent768_PE/resampler/anyres-e2p_mlp/
runs/ADDDATA_S2Q3_1_latent768_PE/finetune/anyres-e2p_mlp/
```

### 2. 체크포인트 파일명
```
{vision_short}_{resampler}_{crop_short}_{dataset}_epoch{XX}_loss{Y.YYYY}.ckpt

예시:
siglip_mlp_anyres-e2p_quic360_epoch03_loss0.4523.ckpt
siglip_mlp_anyres-e2p_quic360_epoch05_loss0.4201.ckpt
```

### 3. WandB Run Name
```
{experiment_name}/{stage}/{vision}_{resampler}_{crop}_{dataset}_{timestamp}

예시:
ADDDATA_S2Q3_1_latent768_PE/vision/siglip_mlp_anyres-e2p_quic360_1015-1430
ADDDATA_S2Q3_1_latent768_PE/resampler/siglip_mlp_anyres-e2p_quic360_1015-1530
```

### 4. Stage State 파일
```
runs/{experiment_name}_stage_state.json

예시:
runs/ADDDATA_S2Q3_1_latent768_PE_stage_state.json
```

### 5. 메타데이터 파일
```
{ckpt_dir}/checkpoint_metadata.json

예시:
runs/ADDDATA_S2Q3_1_latent768_PE/finetune/anyres-e2p_mlp/checkpoint_metadata.json
```

## 이름 구성 요소 (자동 추출)

### Vision Encoder (자동 추출)
```yaml
models:
  vision_name: "google/siglip2-so400m-patch16-256"
  # 기본: vision_short → "siglip2" (첫 토큰)

experiment:
  auto_name: true
  siglip_include_patch_res: true  # 옵션: SigLIP에 한해 세부 버전 표기
  # 결과 예: 실험명에 vision 부분이 "siglip2-so400m_p16_256" 으로 표기
```

### Language Model (자동 추출)
```yaml
models:
  language_model_name: "Qwen/Qwen3-0.6B"
  # → "Qwen3" (자동 추출)
```

### Resampler
```yaml
models:
  resampler_type: "mlp"  # 또는 "qformer", "perceiver", "bimamba"
```

### Crop Strategy (자동 변환)
```yaml
image_processing:
  crop_strategy: "anyres_e2p"
  # → crop_short: "anyres-e2p" (언더스코어를 하이픈으로)
```

### Dataset Name (자동 추출)
```yaml
training:
  stage_configs:
    vision:
      data:
        csv_train: "data/quic360/train.csv"
        # → dataset_name: "train" (파일명에서 추출)
        
        # 여러 CSV의 경우:
        csv_train:
          - "data/quic360/train.csv"
          - "data/stanford/train.csv"
        # → dataset_name: "train_plus1" (첫 번째 이름 + 추가 개수)
```

## YAML 설정 예시

### ✅ 올바른 설정 (권장)
```yaml
experiment:
  name: auto  # 또는 auto_name: true
  siglip_include_patch_res: true  # SigLIP 상세 표기 활성화
  description: "SigLIP + Qwen3 with anyres_e2p"
  
training:
  # prefix 제거됨 - experiment.name 자동 사용
  stages: ["vision", "resampler", "finetune"]
```

### ❌ 이전 방식 (더 이상 필요 없음)
```yaml
experiment:
  name: "ADDDATA_S2Q3_1_latent768_PE"
  
training:
  prefix: "ADDDATA_S2Q3_1_latent768_PE"  # ❌ 중복! 제거하세요
  stages: ["vision", "resampler", "finetune"]
```

## 하위 호환성

기존 코드는 `training.prefix`도 지원하지만, `experiment.name`이 우선합니다:

```python
# train.py 내부 로직
experiment_name = (
    cfg.get("experiment", {}).get("name")           # 1순위 (권장)
    or cfg.get("training", {}).get("prefix")        # 2순위 (하위 호환)
    or "panovlm_exp"                                # 3순위 (기본값)
)
```

## 실전 예시

### 실험 1: SigLIP + MLP
```yaml
experiment:
  name: "EXP1_siglip_mlp"
  
models:
  vision_name: "google/siglip2-so400m-patch16-256"
  resampler_type: "mlp"
  
image_processing:
  crop_strategy: "anyres_e2p"
```

**생성되는 경로:**
```
runs/EXP1_siglip_mlp/vision/anyres-e2p_mlp/
runs/EXP1_siglip_mlp/resampler/anyres-e2p_mlp/
runs/EXP1_siglip_mlp/finetune/anyres-e2p_mlp/
```

### 실험 2: RICE-ViT + BiMamba
```yaml
experiment:
  name: "EXP2_rice_bimamba"
  
models:
  vision_name: "DeepGlint-AI/rice-vit-large-patch14-560"
  resampler_type: "bimamba"
  
image_processing:
  crop_strategy: "sliding_window"
```

**생성되는 경로:**
```
runs/EXP2_rice_bimamba/vision/sliding-window_bimamba/
runs/EXP2_rice_bimamba/resampler/sliding-window_bimamba/
runs/EXP2_rice_bimamba/finetune/sliding-window_bimamba/
```

## 문제 해결

### Q: experiment.name을 변경하면?
A: 모든 경로가 자동으로 새 이름으로 생성됩니다. 기존 체크포인트는 그대로 유지됩니다.

### Q: 이전 training.prefix를 사용한 체크포인트는?
A: 여전히 로드 가능합니다. `resolve_model_dir()`가 자동으로 찾습니다.

### Q: 특정 stage만 다른 이름을 사용하고 싶다면?
A: 불가능합니다. 일관성을 위해 모든 stage가 같은 experiment.name을 사용합니다.

### Q: WandB 프로젝트 이름은?
A: `training.wandb_project`에서 별도로 설정합니다 (experiment.name과 독립적).

## 관련 문서
- [CHECKPOINT_METADATA.md](CHECKPOINT_METADATA.md) - 메타데이터 시스템
- [CHECKPOINT_EVAL_GUIDE.md](CHECKPOINT_EVAL_GUIDE.md) - 평가 시스템
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 전체 설정 가이드
