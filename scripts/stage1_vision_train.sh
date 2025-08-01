#!/bin/bash

# =============================================================================
# Stage 1: Vision Encoder Training
# =============================================================================

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Setup directories
setup_directories

# Print configuration
print_config
echo "Stage: Vision Training"
echo "Batch Size: $VISION_BATCH_SIZE"
echo "Epochs: $VISION_EPOCHS"
echo "========================================"

python train.py \
    --stage vision \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --epochs "${VISION_EPOCHS}" \
    --batch-size "${VISION_BATCH_SIZE}" \
    --resampler "${RESAMPLER}" \
    --crop-strategy "${CROP_STRATEGY}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
    --max-txt-len "${MAX_TXT_LEN}" \
    --image-size ${IMAGE_SIZE} \
    --system-msg "${VISION_SYSTEM_MSG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "vision_${TIMESTAMP}" \
    --vicreg-loss-weight 1.0 \
    2>&1 | tee "logs/vision_${TIMESTAMP}.log"

VISION_CHECKPOINT="runs/${CROP_STRATEGY}_vision_${RESAMPLER}/best.ckpt"

if [ ! -f "$VISION_CHECKPOINT" ]; then
    echo "Error: Vision stage checkpoint not found: $VISION_CHECKPOINT"
    exit 1
fi

echo "Stage 1 completed. Checkpoint: $VISION_CHECKPOINT"