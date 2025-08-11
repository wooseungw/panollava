#!/bin/bash

# =============================================================================
# PanoLLaVA Training Configuration
# =============================================================================

# Exit on any error (다른 스크립트에서 source할 때 필요)
set -e

# GPU 설정
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

# 모델 설정
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen2.5-0.5B"
RESAMPLER="mlp"
CROP_STRATEGY="e2p"

MAX_TEXT_LENGTH=128
IMAGE_SIZE="224 224"

# 데이터 설정
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"

# 학습 설정
NUM_WORKERS=16
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

# Stage별 커스텀 시스템 메시지
VISION_SYSTEM_MSG="You are a helpful assistant that describes panoramic images."
RESAMPLER_SYSTEM_MSG="You are a helpful assistant that understands and describes panoramic images in detail."
FINETUNE_SYSTEM_MSG="You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly."

# LoRA 설정 (finetune 단계용)
USE_LORA=false            # LoRA 사용 여부
LORA_RANK=16              # LoRA rank (낮을수록 적은 파라미터)
LORA_ALPHA=32             # LoRA alpha (일반적으로 rank의 2배)
LORA_DROPOUT=0.1          # LoRA dropout rate
SAVE_LORA_ONLY=false      # LoRA 가중치만 저장할지 여부

# LoRA 타겟 모듈 (Qwen 모델용 기본 설정)
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

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
    echo "Workers: $NUM_WORKERS"
    echo "WandB Project: $WANDB_PROJECT"
    echo "========================================"
}

# 특정 설정 오버라이드 함수
override_config() {
    local param_name="$1"
    local param_value="$2"
    
    case "$param_name" in
        "vision_model"|"vision-model")
            VISION_MODEL="$param_value"
            ;;
        "lm_model"|"lm-model")
            LM_MODEL="$param_value"
            ;;
        "resampler")
            RESAMPLER="$param_value"
            ;;
        "crop_strategy"|"crop-strategy")
            CROP_STRATEGY="$param_value"
            ;;
        "workers")
            NUM_WORKERS="$param_value"
            ;;
        "wandb_project"|"wandb-project")
            WANDB_PROJECT="$param_value"
            ;;
        "max_text_length"|"max-text-length")  # 추가
            MAX_TEXT_LENGTH="$param_value"
            ;;
        "finetune_system_msg"|"finetune-system-msg")  # 추가
            FINETUNE_SYSTEM_MSG="$param_value"
            ;;
        "vision_system_msg"|"vision-system-msg")  # 추가
            VISION_SYSTEM_MSG="$param_value"
            ;;
        "resampler_system_msg"|"resampler-system-msg")  # 추가
            RESAMPLER_SYSTEM_MSG="$param_value"
            ;;
        # LoRA 설정들
        "use_lora"|"use-lora")
            USE_LORA="$param_value"
            ;;
        "lora_rank"|"lora-rank")
            LORA_RANK="$param_value"
            ;;
        "lora_alpha"|"lora-alpha")
            LORA_ALPHA="$param_value"
            ;;
        "lora_dropout"|"lora-dropout")
            LORA_DROPOUT="$param_value"
            ;;
        "save_lora_only"|"save-lora-only")
            SAVE_LORA_ONLY="$param_value"
            ;;
        "lora_target_modules"|"lora-target-modules")
            LORA_TARGET_MODULES="$param_value"
            ;;
        *)
            echo "Warning: Unknown configuration parameter: $param_name"
            ;;
    esac
}