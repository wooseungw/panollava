#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
# =============================================================================
# Stage 2: Resampler Training
# =============================================================================

set -e  # Exit on any error

echo "========================================"
echo "Stage 2: Resampler Training"
echo "========================================"

# Configuration
STAGE="resampler"
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
VICREG_LOSS_WEIGHT=0.0
MAX_TXT_LEN=64

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
WANDB_NAME="stage2_resampler_$(date +%Y%m%d_%H%M%S)"

# Checkpoint Configuration
# Set this to the best checkpoint from Stage 1
RESUME_FROM="${1:-runs/vlm_vision/checkpoints/best.ckpt}"

if [ ! -f "$RESUME_FROM" ]; then
    echo "Error: Checkpoint file not found: $RESUME_FROM"
    echo "Usage: $0 <path_to_stage1_checkpoint>"
    echo "Example: $0 runs/vlm_vision/checkpoints/epoch=02-val_loss=0.123.ckpt"
    exit 1
fi

echo "Resuming from checkpoint: $RESUME_FROM"

# Create directories
mkdir -p logs
mkdir -p runs/vlm_resampler/checkpoints

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
    --num-workers "${NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --resume-from "${RESUME_FROM}" \
    2>&1 | tee "logs/stage2_resampler_$(date +%Y%m%d_%H%M%S).log"

echo "Stage 2 training completed!"
echo "Best checkpoint saved in: runs/vlm_resampler/checkpoints/"
echo "Next: Run stage3_finetune.sh with the checkpoint path"