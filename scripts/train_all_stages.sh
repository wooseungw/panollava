#!/bin/bash

# =============================================================================
# PanoLLaVA Full 3-Stage Training Pipeline
# =============================================================================

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================"
echo "PanoLLaVA Full Training Pipeline"
echo "========================================"

print_config
setup_directories

# Validate data files
if [ ! -f "$CSV_TRAIN" ]; then
    echo "Error: Training data not found: $CSV_TRAIN"
    exit 1
fi

if [ ! -f "$CSV_VAL" ]; then
    echo "Error: Validation data not found: $CSV_VAL"
    exit 1
fi

TIMESTAMP=$(generate_timestamp)

# =============================================================================
# Stage 1: Vision Training (VICReg)
# =============================================================================
echo ""
echo "========================================="
echo "Stage 1/3: Vision Training"
echo "========================================="

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
    2>&1 | tee "${LOG_DIR}/${PREFIX}_vision_${TIMESTAMP}.log"

VISION_CHECKPOINT="runs/${PREFIX}_${CROP_STRATEGY}_vision_${RESAMPLER}/best.ckpt"
if [ ! -f "$VISION_CHECKPOINT" ]; then
    echo "Error: Vision checkpoint not found: $VISION_CHECKPOINT"
    exit 1
fi
echo "✓ Stage 1 completed: $VISION_CHECKPOINT"

# =============================================================================
# Stage 2: Resampler Training  
# =============================================================================
echo ""
echo "========================================="
echo "Stage 2/3: Resampler Training"
echo "========================================="

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
    --prefix "$PREFIX" \
    2>&1 | tee "${LOG_DIR}/${PREFIX}_resampler_${TIMESTAMP}.log"

RESAMPLER_CHECKPOINT="runs/${PREFIX}_${CROP_STRATEGY}_resampler_${RESAMPLER}/best.ckpt"
if [ ! -f "$RESAMPLER_CHECKPOINT" ]; then
    echo "Error: Resampler checkpoint not found: $RESAMPLER_CHECKPOINT"
    exit 1
fi
echo "✓ Stage 2 completed: $RESAMPLER_CHECKPOINT"

# =============================================================================
# Stage 3: Full Model Fine-tuning
# =============================================================================
echo ""
echo "========================================="
echo "Stage 3/3: Fine-tuning"
echo "========================================="

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
    2>&1 | tee "${LOG_DIR}/${PREFIX}_finetune_${TIMESTAMP}.log"

FINAL_CHECKPOINT="runs/${PREFIX}_${CROP_STRATEGY}_finetune_${RESAMPLER}/best.ckpt"
if [ ! -f "$FINAL_CHECKPOINT" ]; then
    echo "Error: Final checkpoint not found: $FINAL_CHECKPOINT"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ All stages completed successfully!"
echo "========================================"
echo "Logs:"
echo "  Stage 1: ${LOG_DIR}/${PREFIX}_vision_${TIMESTAMP}.log"
echo "  Stage 2: ${LOG_DIR}/${PREFIX}_resampler_${TIMESTAMP}.log"  
echo "  Stage 3: ${LOG_DIR}/${PREFIX}_finetune_${TIMESTAMP}.log"
echo ""
echo "Final model: $FINAL_CHECKPOINT"
echo "========================================"
