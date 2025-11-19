# 체크포인트 기반 평가 가이드

## 개요

PanoLLaVA는 **체크포인트 디렉토리 기반 평가**를 지원합니다. 이를 통해 복잡한 config 파일 없이 간단하게 모델을 평가할 수 있습니다.

## 핵심 기능

### 1. 메타데이터 자동 로드
훈련 시 저장된 `checkpoint_metadata.json`을 자동으로 읽어 모든 설정을 복원합니다:
- 모델 아키텍처 (vision encoder, language model, resampler)
- 이미지 처리 설정 (crop_strategy, image_size, fov_deg 등)
- 하이퍼파라미터 (learning_rate, batch_size 등)
- 데이터셋 정보

### 2. 스마트 체크포인트 선택
우선순위에 따라 자동으로 체크포인트를 선택합니다:
1. **best.ckpt** (심볼릭 링크) - 가장 낮은 validation loss
2. **last.ckpt** (심볼릭 링크) - 마지막 epoch
3. **최신 .ckpt 파일** - 수정 시간 기준

### 3. LoRA 가중치 자동 탐색
체크포인트 디렉토리 내 `lora_weights/` 폴더를 자동으로 찾아 로드합니다.

## 사용 방법

### 방법 1: 체크포인트 디렉토리 지정 (권장 ⭐)

```bash
# 가장 간단한 방법 - 메타데이터에서 모든 설정 자동 로드
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/

# CSV 파일 명시 (메타데이터의 데이터셋 경로 대신 사용)
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \
    --csv-input data/quic360/test.csv

# 샘플 수 제한 (빠른 테스트)
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \
    --csv-input data/quic360/test.csv \
    --max-samples 100
```

### 방법 2: Config 기반 평가 (기존 방식)

```bash
# config에서 자동으로 체크포인트 탐색
python scripts/eval.py \
    --config configs/default.yaml \
    --csv-input data/quic360/test.csv
```

### 방법 3: 하이브리드 (체크포인트 + Config)

```bash
# 체크포인트는 명시, 나머지는 config 사용
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \
    --config configs/default.yaml \
    --csv-input data/quic360/test.csv
```

## 체크포인트 디렉토리 구조

올바른 평가를 위해 다음 구조가 권장됩니다:

```
runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/
├── checkpoint_metadata.json          # 필수: 모델 설정 정보
├── best.ckpt -> siglip_mlp_...ckpt  # 권장: 최고 성능 체크포인트 링크
├── last.ckpt -> siglip_mlp_...ckpt  # 권장: 마지막 체크포인트 링크
├── siglip_mlp_anyres-e2p_quic360_epoch03_loss0.4523.ckpt
├── siglip_mlp_anyres-e2p_quic360_epoch05_loss0.4201.ckpt
└── lora_weights/                     # 선택: LoRA 가중치
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### 메타데이터 파일 예시

`checkpoint_metadata.json`:
```json
{
  "experiment_name": "ADDDATA_SQ3_1",
  "stage": "finetune",
  "model_config": {
    "vision_name": "google/siglip2-so400m-patch14-224",
    "language_model_name": "Qwen/Qwen3-0.6B",
    "resampler_type": "mlp",
    "latent_dimension": 768,
    "image_size": [224, 224]
  },
  "training_config": {
    "crop_strategy": "anyres_e2p",
    "learning_rate": 0.0001,
    "batch_size": 16,
    "fov_deg": 90.0,
    "use_vision_processor": true
  },
  "dataset": {
    "train_csv": "data/quic360/train.csv",
    "val_csv": "data/quic360/val.csv",
    "dataset_name": "quic360"
  }
}
```

## 로그 출력 예시

```
============================================================
📂 체크포인트 디렉토리: runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp
============================================================
✅ 메타데이터 로드 성공: runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/checkpoint_metadata.json
✅ Using best checkpoint: runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/siglip_mlp_anyres-e2p_quic360_epoch05_loss0.4201.ckpt
============================================================
📋 메타데이터에서 로드된 정보:
  - Experiment: ADDDATA_SQ3_1
  - Stage: finetune
  - Vision: google/siglip2-so400m-patch14-224
  - Language: Qwen/Qwen3-0.6B
  - Resampler: mlp
  - Crop Strategy: anyres_e2p
============================================================
✅ 메타데이터를 config에 병합 완료
✅ Auto-found LoRA weights: runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/lora_weights
```

## 우선순위 규칙

설정값의 우선순위는 다음과 같습니다:

1. **CLI 인자** (가장 높음)
   - `--csv-input`, `--max-samples` 등

2. **체크포인트 메타데이터**
   - `checkpoint_metadata.json`의 모델/훈련 설정

3. **Config 파일**
   - `--config`로 지정된 YAML 파일

4. **기본값** (가장 낮음)
   - 코드에 하드코딩된 fallback 값

## 문제 해결

### 메타데이터 파일이 없는 경우

```
⚠️ 메타데이터 파일을 찾을 수 없습니다: .../checkpoint_metadata.json
```

**해결책**: `--config` 옵션으로 설정 파일을 명시하거나, 메타데이터가 있는 최신 체크포인트를 사용하세요.

### 체크포인트 파일을 찾을 수 없는 경우

```
FileNotFoundError: 체크포인트 파일을 찾을 수 없습니다: ...
```

**해결책**: 
- 디렉토리에 `.ckpt` 파일이 있는지 확인
- `best.ckpt` 또는 `last.ckpt` 심볼릭 링크가 올바른지 확인
- 전체 경로가 정확한지 확인

### 설정 불일치 경고

```
⚠️ Config와 메타데이터의 설정이 다릅니다
```

**해결책**: 메타데이터가 우선되므로 보통 무시해도 됩니다. 명시적으로 설정을 변경하려면 `--config`를 제거하세요.

## 모범 사례

### ✅ DO

1. **체크포인트 디렉토리만 지정** (메타데이터 자동 로드)
   ```bash
   python scripts/eval.py --checkpoint-dir runs/my_experiment/finetune/anyres-e2p_mlp/
   ```

2. **평가용 CSV만 명시** (다른 데이터셋 테스트 시)
   ```bash
   python scripts/eval.py \
       --checkpoint-dir runs/my_experiment/finetune/anyres-e2p_mlp/ \
       --csv-input data/new_test_set.csv
   ```

3. **상세 로깅 활성화** (디버깅 시)
   ```bash
   python scripts/eval.py \
       --checkpoint-dir runs/my_experiment/finetune/anyres-e2p_mlp/ \
       --log-samples \
       --log-interval 10
   ```

### ❌ DON'T

1. **Config와 체크포인트를 동시에 지정하면서 설정 충돌**
   ```bash
   # 혼란 야기 - 메타데이터와 config 설정이 다를 수 있음
   python scripts/eval.py \
       --config configs/different_model.yaml \
       --checkpoint-dir runs/my_experiment/
   ```

2. **메타데이터 없이 체크포인트만 지정**
   ```bash
   # checkpoint_metadata.json이 없으면 설정 불완전
   python scripts/eval.py --checkpoint-dir runs/old_experiment/
   ```

## 추가 옵션

### 샘플 로깅

```bash
# 배치별 예측/정답 텍스트 출력
python scripts/eval.py \
    --checkpoint-dir runs/my_experiment/finetune/anyres-e2p_mlp/ \
    --log-samples \
    --log-interval 25 \
    --log-max-samples 50
```

- `--log-samples`: 샘플별 로그 활성화
- `--log-interval N`: N 배치마다 로그 출력
- `--log-max-samples M`: 최대 M개 샘플까지만 로그

### 샘플 수 제한

```bash
# 빠른 테스트 (100개 샘플만)
python scripts/eval.py \
    --checkpoint-dir runs/my_experiment/finetune/anyres-e2p_mlp/ \
    --max-samples 100
```

## 관련 문서

- [CHECKPOINT_METADATA.md](CHECKPOINT_METADATA.md) - 메타데이터 시스템 설계
- [NAMING_CONVENTION.md](NAMING_CONVENTION.md) - 체크포인트 파일명 규칙
- [VLM_EVALUATION_GUIDE.md](VLM_EVALUATION_GUIDE.md) - 평가 메트릭 상세 가이드
