#!/bin/bash

# =============================================================================
# Full 3-Stage Training Pipeline
# =============================================================================

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================"
echo "PanoLLaVA Full Training Pipeline"
echo "========================================"

# Print configuration
print_config

# Setup directories
setup_directories

# Validate data files
if [ ! -f "$CSV_TRAIN" ]; then
    echo "Error: Training data file not found: $CSV_TRAIN"
    exit 1
fi

if [ ! -f "$CSV_VAL" ]; then
    echo "Error: Validation data file not found: $CSV_VAL"
    exit 1
fi

echo "Starting full 3-stage training pipeline..."
echo "Training data: $CSV_TRAIN"
echo "Validation data: $CSV_VAL"

TIMESTAMP=$(generate_timestamp)

# =============================================================================
# Stage 1: Vision Training
# =============================================================================
echo ""
echo "========================================="
echo "Stage 1/3: Vision Training (VICReg)"
echo "========================================="

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
    --max-text-length "${MAX_TEXT_LENGTH}" \
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

# =============================================================================
# Stage 2: Resampler Training
# =============================================================================
echo ""
echo "========================================="
echo "Stage 2/3: Resampler Training"
echo "========================================="

python train.py \
    --stage resampler \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --resampler "${RESAMPLER}" \
    --epochs 1 \
    --batch-size 4 \
    --crop-strategy "${CROP_STRATEGY}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
    --max-text-length "${MAX_TEXT_LENGTH}" \
    --image-size ${IMAGE_SIZE} \
    --system-msg "${RESAMPLER_SYSTEM_MSG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "resampler_${TIMESTAMP}" \
    --vicreg-loss-weight 0.0 \
    --resume-from "${VISION_CHECKPOINT}" \
    2>&1 | tee "logs/resampler_${TIMESTAMP}.log"

RESAMPLER_CHECKPOINT="runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/best.ckpt"

if [ ! -f "$RESAMPLER_CHECKPOINT" ]; then
    echo "Error: Resampler stage checkpoint not found: $RESAMPLER_CHECKPOINT"
    exit 1
fi

echo "Stage 2 completed. Checkpoint: $RESAMPLER_CHECKPOINT"

# =============================================================================
# Stage 3: Full Model Fine-tuning
# =============================================================================
echo ""
echo "========================================="
echo "Stage 3/3: Full Model Fine-tuning"
echo "========================================="

# finetune 단계에서 LoRA 설정 추가
FINETUNE_ARGS=""
if [ "$USE_LORA" = "true" ]; then
    FINETUNE_ARGS="--use-lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DROPOUT"
    if [ "$SAVE_LORA_ONLY" = "true" ]; then
        FINETUNE_ARGS="$FINETUNE_ARGS --save-lora-only"
    fi
    if [ -n "$LORA_TARGET_MODULES" ]; then
        FINETUNE_ARGS="$FINETUNE_ARGS --lora-target-modules $LORA_TARGET_MODULES"
    fi
fi

python train.py \
    --stage finetune \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --resampler "${RESAMPLER}" \
    --epochs 1 \
    --batch-size 4 \
    --crop-strategy "${CROP_STRATEGY}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
    --max-text-length "${MAX_TEXT_LENGTH}" \
    --image-size ${IMAGE_SIZE} \
    --system-msg "${FINETUNE_SYSTEM_MSG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "finetune_${TIMESTAMP}" \
    --resume-from "${RESAMPLER_CHECKPOINT}" \
    $FINETUNE_ARGS \
    2>&1 | tee "logs/finetune_${TIMESTAMP}.log"

FINAL_CHECKPOINT="runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/best.ckpt"

if [ ! -f "$FINAL_CHECKPOINT" ]; then
    echo "Error: Final stage checkpoint not found: $FINAL_CHECKPOINT"
    exit 1
fi

echo "========================================"
echo "Full training pipeline completed!"
echo "========================================"
echo "Stage 1 (Vision): logs/vision_${TIMESTAMP}.log"
echo "Stage 2 (Resampler): logs/resampler_${TIMESTAMP}.log"
echo "Stage 3 (Finetune): logs/finetune_${TIMESTAMP}.log"
echo ""
echo "Final model: $FINAL_CHECKPOINT"
echo "========================================"
