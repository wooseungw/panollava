#!/bin/bash

# =============================================================================
# Fine-tuning Stage Training (Stage 3/3)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================="
echo "Fine-tuning Training"
echo "========================================="

print_config
setup_directories

TIMESTAMP=$(generate_timestamp)

# Check for resampler checkpoint
RESAMPLER_CHECKPOINT="runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/best.ckpt"
if [ ! -f "$RESAMPLER_CHECKPOINT" ]; then
    echo "Error: Resampler checkpoint not found: $RESAMPLER_CHECKPOINT"
    echo "Please run train_resampler.sh first"
    exit 1
fi

# Build LoRA arguments
LORA_ARGS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_ARGS="--use-lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DROPOUT --lora-target-modules $LORA_TARGET_MODULES"
    if [ "$SAVE_LORA_ONLY" = "true" ]; then
        LORA_ARGS="$LORA_ARGS --save-lora-only"
    fi
    echo "LoRA enabled: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
fi

python train.py \
    --stage finetune \
    --vision-name "$VISION_MODEL" \
    --lm-name "$LM_MODEL" \
    --resampler "$RESAMPLER" \
    --crop-strategy "$CROP_STRATEGY" \
    --csv-train "$CSV_TRAIN" \
    --csv-val "$CSV_VAL" \
    --image-size $IMAGE_SIZE \
    --max-text-length $MAX_TEXT_LENGTH \
    --epochs $FINETUNE_EPOCHS \
    --batch-size $FINETUNE_BATCH_SIZE \
    --lr $FINETUNE_LR \
    --vicreg-loss-weight 0.0 \
    --overlap-ratio $OVERLAP_RATIO \
    --num-workers $NUM_WORKERS \
    --system-msg "$SYSTEM_MSG_FINETUNE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "finetune_${TIMESTAMP}" \
    --resume-from "$RESAMPLER_CHECKPOINT" \
    --prefix "$PREFIX" \
    $LORA_ARGS \
    2>&1 | tee "${LOG_DIR}/finetune_${TIMESTAMP}.log"

echo "✓ Fine-tuning completed"
echo "Checkpoint: runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/best.ckpt"
echo "Log: ${LOG_DIR}/finetune_${TIMESTAMP}.log"
