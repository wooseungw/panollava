#!/bin/bash

# =============================================================================
# PanoLLaVA Training Configuration
# =============================================================================

set -e

# =============================================================================
# Environment & GPU Settings
# =============================================================================
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
export WANDB_PROJECT="panollava-training"

# =============================================================================
# Model Configuration
# =============================================================================
VISION_MODEL="google/siglip2-base-patch16-224"
LM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESAMPLER="mlp"
#PREIX Must be Changed.
# =============================================================================
# Data Configuration
# =============================================================================
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"
CROP_STRATEGY="e2p"
IMAGE_SIZE="224 224"
MAX_TEXT_LENGTH=256
OVERLAP_RATIO=0.5

# =============================================================================
# Training Configuration
# =============================================================================
NUM_WORKERS=16

# Stage-specific settings with Learning Rates
VISION_EPOCHS=3
VISION_BATCH_SIZE=16
VISION_LR="4e-5"
VISION_VICREG_LOSS_WEIGHT=1.0

RESAMPLER_EPOCHS=1
RESAMPLER_BATCH_SIZE=8
RESAMPLER_LR="2e-5"
RESAMPLER_VICREG_LOSS_WEIGHT=0.2

FINETUNE_EPOCHS=1
FINETUNE_BATCH_SIZE=8
FINETUNE_LR="1e-5"

# =============================================================================
# LoRA Configuration (finetune only)
# =============================================================================
USE_LORA=true
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
SAVE_LORA_ONLY=false
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

# =============================================================================
# System Messages
# =============================================================================
SYSTEM_MSG_DEFAULT="You are a helpful assistant."
SYSTEM_MSG_FINETUNE="You are a helpful assistant. When you analyze a panoramic view, answer to the user's specific query first, then briefly justify with key visual evidence."

# =============================================================================
# Directory Configuration
# =============================================================================
LOG_DIR="logs"
RUNS_DIR="runs"
PREFIX="siglipv2qwen25Instruct"
# =============================================================================
# Utility Functions
# =============================================================================

generate_timestamp() {
    echo $(date +%Y%m%d_%H%M%S)
}

setup_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$RUNS_DIR"
}

print_config() {
    echo "========================================"
    echo "PanoLLaVA Training Configuration"
    echo "========================================"
    echo "Vision Model: $VISION_MODEL"
    echo "Language Model: $LM_MODEL"
    echo "Resampler: $RESAMPLER"
    echo "Crop Strategy: $CROP_STRATEGY"
    echo ""
    echo "Learning Rates:"
    echo "  Vision:    $VISION_LR"
    echo "  Resampler: $RESAMPLER_LR"
    echo "  Finetune:  $FINETUNE_LR"
    echo ""
    echo "Batch Sizes & Epochs:"
    echo "  Vision:    ${VISION_BATCH_SIZE} batch, ${VISION_EPOCHS} epochs"
    echo "  Resampler: ${RESAMPLER_BATCH_SIZE} batch, ${RESAMPLER_EPOCHS} epochs"  
    echo "  Finetune:  ${FINETUNE_BATCH_SIZE} batch, ${FINETUNE_EPOCHS} epochs"
    echo ""
    echo "Training Data: $CSV_TRAIN"
    echo "Validation Data: $CSV_VAL"
    echo "Workers: $NUM_WORKERS"
    echo "WandB Project: $WANDB_PROJECT"
    echo "========================================"
}
