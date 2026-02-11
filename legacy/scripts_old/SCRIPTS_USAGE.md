# PanoLLaVA Training Scripts

간단하고 깔끔하게 정리된 3단계 훈련 스크립트입니다.

## 파일 구조

```
scripts/
├── config.sh              # 모든 설정 변수 (여기서만 수정하면 됨!)
├── train_all_stages.sh     # 전체 3단계 훈련 (한번에 실행)
├── train_vision.sh         # 1단계: Vision 훈련
├── train_resampler.sh      # 2단계: Resampler 훈련  
├── train_finetune.sh       # 3단계: Fine-tuning
└── SCRIPTS_USAGE.md        # 이 파일
```

## 사용법

### 1. 설정 수정
`config.sh`에서 원하는 값들을 수정하세요:

```bash
# 모델 변경
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen2.5-0.5B"

# 데이터 경로
CSV_TRAIN="data/quic360/train.csv" 
CSV_VAL="data/quic360/valid.csv"

# 배치 크기 및 에포크 조정
VISION_BATCH_SIZE=16
RESAMPLER_BATCH_SIZE=8
FINETUNE_BATCH_SIZE=8

# LoRA 설정
USE_LORA=true
LORA_RANK=16
```

### 2. 훈련 실행

**전체 3단계 한번에 실행:**
```bash
./scripts/train_all_stages.sh
```

**개별 단계 실행:**
```bash
# 1단계만
./scripts/train_vision.sh

# 2단계만 (1단계 완료 후)
./scripts/train_resampler.sh  

# 3단계만 (2단계 완료 후)
./scripts/train_finetune.sh
```

## 주요 특징

- **변수명 통일**: 모든 스크립트에서 일관된 변수명 사용
- **중앙집중 설정**: `config.sh`에서만 설정 수정
- **깔끔한 로그**: 각 단계별로 별도 로그 파일 생성
- **체크포인트 연결**: 이전 단계 체크포인트 자동 연결
- **에러 처리**: 체크포인트 누락 시 친절한 에러 메시지
- **진행상황 표시**: 각 단계 완료 시 ✓ 표시

## 체크포인트 경로

```
runs/
├── e2p_vision_mlp/best.ckpt      # Vision 단계
├── e2p_resampler_mlp/best.ckpt   # Resampler 단계  
└── e2p_finetune_mlp/best.ckpt    # Fine-tuning 단계
```

## 로그 파일

```
logs/
├── vision_YYYYMMDD_HHMMSS.log
├── resampler_YYYYMMDD_HHMMSS.log
└── finetune_YYYYMMDD_HHMMSS.log
```
