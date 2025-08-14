#!/bin/bash

# =============================================================================
# Stage 3: Finetune Training
# =============================================================================

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Setup directories
setup_directories

# Print configuration
print_config
echo "Stage: Finetune Training"
echo "Batch Size: $FINETUNE_BATCH_SIZE"
echo "Epochs: $FINETUNE_EPOCHS"
echo "========================================"

RESAMPLER_CHECKPOINT="${RESAMPLER_CHECKPOINT_DIR}/best.ckpt"

python train.py \
    --stage finetune \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --resampler "${RESAMPLER}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --batch-size "${FINETUNE_BATCH_SIZE}" \
    --crop-strategy "${CROP_STRATEGY}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
    --max-text-length "${MAX_TEXT_LENGTH}" \
    --image-size ${IMAGE_SIZE} \
    --system-msg "${FINETUNE_SYSTEM_MSG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "finetune_${TIMESTAMP}" \
    --resume-from "$RESAMPLER_CHECKPOINT" \
    --prefix "$PREFIX" \
    2>&1 | tee "logs/finetune_${TIMESTAMP}.log"

FINAL_CHECKPOINT="runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/best.ckpt"

if [ ! -f "$FINAL_CHECKPOINT" ]; then
    echo "Error: Final stage checkpoint not found: $FINAL_CHECKPOINT"
    exit 1
fi

echo "Stage 3 completed. Checkpoint: $FINAL_CHECKPOINT"