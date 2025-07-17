#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
# =============================================================================
# Stage 3: End-to-End Fine-tuning
# =============================================================================

set -e  # Exit on any error

echo "========================================"
echo "Stage 3: End-to-End Fine-tuning"
echo "========================================"

# Configuration
STAGE="finetune"
EPOCHS=3
BATCH_SIZE=2
LEARNING_RATE=1e-6
VICREG_LOSS_WEIGHT=0.0
MAX_TXT_LEN=256

# Model Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"

# Data Configuration
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"

# Training Configuration
NUM_WORKERS=64
WANDB_PROJECT="panollava-training"
WANDB_NAME="stage3_finetune_$(date +%Y%m%d_%H%M%S)"

RESUME_FROM="${1:-runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/resampler/best.ckpt}"

if [ ! -f "$RESUME_FROM" ]; then
    echo "Error: Checkpoint file not found: $RESUME_FROM"
    echo "Usage: $0 <path_to_stage2_checkpoint>"
    echo "Example: $0 runs/vlm_resampler/checkpoints/epoch=04-val_loss=0.089.ckpt"
    exit 1
fi

echo "Resuming from checkpoint: $RESUME_FROM"

# Create directories
mkdir -p logs
mkdir -p runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/finetune

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
    --resume-from "${RESUME_FROM}" \
    2>&1 | tee "logs/stage3_finetune_$(date +%Y%m%d_%H%M%S).log"

echo "Stage 3 training completed!"
echo "Best checkpoint saved in: runs/vlm_finetune/checkpoints/"
echo "Final model available at: runs/vlm_finetune/model_final.safetensors"