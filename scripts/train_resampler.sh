#!/bin/bash

# =============================================================================
# Resampler Stage Training (Stage 2/3)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================="
echo "Resampler Training"
echo "========================================="

print_config
setup_directories

TIMESTAMP=$(generate_timestamp)

# Check for vision checkpoint
VISION_CHECKPOINT="runs/${CROP_STRATEGY}_vision_${RESAMPLER}/best.ckpt"
if [ ! -f "$VISION_CHECKPOINT" ]; then
    echo "Error: Vision checkpoint not found: $VISION_CHECKPOINT"
    echo "Please run train_vision.sh first"
    exit 1
fi

python train.py \
    --stage resampler \
    --vision-name "$VISION_MODEL" \
    --lm-name "$LM_MODEL" \
    --resampler "$RESAMPLER" \
    --crop-strategy "$CROP_STRATEGY" \
    --csv-train "$CSV_TRAIN" \
    --csv-val "$CSV_VAL" \
    --image-size $IMAGE_SIZE \
    --max-text-length $MAX_TEXT_LENGTH \
    --epochs $RESAMPLER_EPOCHS \
    --batch-size $RESAMPLER_BATCH_SIZE \
    --lr $RESAMPLER_LR \
    --vicreg-loss-weight 0.0 \
    --overlap-ratio $OVERLAP_RATIO \
    --num-workers $NUM_WORKERS \
    --system-msg "$SYSTEM_MSG_DEFAULT" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "resampler_${TIMESTAMP}" \
    --resume-from "$VISION_CHECKPOINT" \
    2>&1 | tee "${LOG_DIR}/resampler_${TIMESTAMP}.log"

echo "✓ Resampler training completed"
echo "Checkpoint: runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/best.ckpt"
echo "Log: ${LOG_DIR}/resampler_${TIMESTAMP}.log"
