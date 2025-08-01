#!/bin/bash

# =============================================================================
# Stage 2: Resampler Training
# =============================================================================

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Setup directories
setup_directories

# Print configuration
print_config
echo "Stage: Resampler Training"
echo "Batch Size: $RESAMPLER_BATCH_SIZE"
echo "Epochs: $RESAMPLER_EPOCHS"
echo "========================================"

VISION_CHECKPOINT="${VISION_CHECKPOINT_DIR}/best.ckpt"

python train.py \
    --stage resampler \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --resampler "${RESAMPLER}" \
    --epochs "${RESAMPLER_EPOCHS}" \
    --batch-size "${RESAMPLER_BATCH_SIZE}" \
    --crop-strategy "${CROP_STRATEGY}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
    --max-txt-len "${MAX_TXT_LEN}" \
    --image-size ${IMAGE_SIZE} \
    --system-msg "${RESAMPLER_SYSTEM_MSG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "resampler_${TIMESTAMP}" \
    --vicreg-loss-weight 0.0 \
    --resume-from "$VISION_CHECKPOINT" \
    2>&1 | tee "logs/resampler_${TIMESTAMP}.log"

RESAMPLER_CHECKPOINT="runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/best.ckpt"

if [ ! -f "$RESAMPLER_CHECKPOINT" ]; then
    echo "Error: Resampler stage checkpoint not found: $RESAMPLER_CHECKPOINT"
    exit 1
fi

echo "Stage 2 completed. Checkpoint: $RESAMPLER_CHECKPOINT"