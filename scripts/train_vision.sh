#!/bin/bash

# =============================================================================
# Vision Stage Training (Stage 1/3)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================="
echo "Vision Training (VICReg Loss)"
echo "========================================="

print_config
setup_directories

TIMESTAMP=$(generate_timestamp)

python train.py \
    --stage vision \
    --vision-name "$VISION_MODEL" \
    --lm-name "$LM_MODEL" \
    --resampler "$RESAMPLER" \
    --crop-strategy "$CROP_STRATEGY" \
    --csv-train "$CSV_TRAIN" \
    --csv-val "$CSV_VAL" \
    --image-size $IMAGE_SIZE \
    --max-text-length $MAX_TEXT_LENGTH \
    --epochs $VISION_EPOCHS \
    --batch-size $VISION_BATCH_SIZE \
    --lr $VISION_LR \
    --vicreg-loss-weight $VICREG_LOSS_WEIGHT \
    --overlap-ratio $OVERLAP_RATIO \
    --num-workers $NUM_WORKERS \
    --system-msg "$SYSTEM_MSG_DEFAULT" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "vision_${TIMESTAMP}" \
    --prefix "$PREFIX" \
    2>&1 | tee "${LOG_DIR}/vision_${TIMESTAMP}.log"

echo "✓ Vision training completed"
echo "Checkpoint: runs/${CROP_STRATEGY}_vision_${RESAMPLER}/best.ckpt"
echo "Log: ${LOG_DIR}/vision_${TIMESTAMP}.log"
