#!/bin/bash

# =============================================================================
# PanoLLaVA Training Configuration
# =============================================================================

# GPU 설정
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

# 모델 설정
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"
CROP_STRATEGY="e2p"

# 데이터 설정
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"

# 학습 설정
NUM_WORKERS=64
WANDB_PROJECT="panollava-training"

# Stage별 배치 크기 및 에포크
VISION_BATCH_SIZE=16
VISION_EPOCHS=3

RESAMPLER_BATCH_SIZE=4
RESAMPLER_EPOCHS=1

FINETUNE_BATCH_SIZE=4
FINETUNE_EPOCHS=1

# 생성 설정 (평가용)
MAX_NEW_TOKENS=64
TEMPERATURE=0.7

# 디렉토리 설정
LOG_DIR="logs"
RUNS_DIR="runs"
EVAL_OUTPUT_DIR="eval_results"

# 체크포인트 경로 템플릿
VISION_CHECKPOINT_DIR="runs/${CROP_STRATEGY}_vision_${RESAMPLER}"
RESAMPLER_CHECKPOINT_DIR="runs/${CROP_STRATEGY}_resampler_${RESAMPLER}"
FINETUNE_CHECKPOINT_DIR="runs/${CROP_STRATEGY}_finetune_${RESAMPLER}"

# 타임스탬프 생성 함수
generate_timestamp() {
    echo $(date +%Y%m%d_%H%M%S)
}

# 디렉토리 생성 함수
setup_directories() {
    mkdir -p $LOG_DIR
    mkdir -p $RUNS_DIR
    mkdir -p $VISION_CHECKPOINT_DIR
    mkdir -p $RESAMPLER_CHECKPOINT_DIR
    mkdir -p $FINETUNE_CHECKPOINT_DIR
    mkdir -p $EVAL_OUTPUT_DIR
}

# 설정 출력 함수
print_config() {
    echo "========================================"
    echo "PanoLLaVA Configuration"
    echo "========================================"
    echo "Vision Model: $VISION_MODEL"
    echo "Language Model: $LM_MODEL"
    echo "Resampler: $RESAMPLER"
    echo "Crop Strategy: $CROP_STRATEGY"
    echo "Training Data: $CSV_TRAIN"
    echo "Validation Data: $CSV_VAL"
    echo "========================================"
}