#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
# =============================================================================
# Stage 1: Vision Encoder Training with VICReg Loss
# =============================================================================

set -e  # Exit on any error

echo "========================================"
echo "Stage 1: Vision Encoder Training"
echo "========================================"

# Configuration
STAGE="vision"
EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=5e-6
VICREG_LOSS_WEIGHT=1.0
MAX_TXT_LEN=32

# Model Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen2.5-0.5B"
RESAMPLER="mlp"

# Data Configuration
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"

# Training Configuration
NUM_WORKERS=4
WANDB_PROJECT="panollava-training"
WANDB_NAME="stage1_vision_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs
mkdir -p runs/${CROP_STRATEGY}_vision_${RESAMPLER}/vision

# Run training
python train.py \
    --stage "${STAGE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --vicreg-loss-weight "${VICREG_LOSS_WEIGHT}" \
    --max-txt-len "${MAX_TXT_LEN}" \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --resampler "${RESAMPLER}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --crop-strategy e2p \
    --num-workers "${NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    2>&1 | tee "logs/stage1_vision_$(date +%Y%m%d_%H%M%S).log"

echo "Stage 1 training completed!"
echo "Best checkpoint saved in: runs/vlm_vision/checkpoints/"
echo "Next: Run stage2_resampler.sh with the checkpoint path"