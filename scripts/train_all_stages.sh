#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
# =============================================================================
# Full 3-Stage Training Pipeline
# =============================================================================

set -e  # Exit on any error

echo "========================================"
echo "PanoLLaVA Full Training Pipeline"
echo "========================================"

# Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen2.5-0.5B"
RESAMPLER="mlp"

# Data Configuration
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"

# Training Configuration
NUM_WORKERS=64
WANDB_PROJECT="panollava-training"

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

# Create directories
mkdir -p logs
mkdir -p runs

# Run all stages using the unified training script
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WANDB_NAME="full_pipeline_${TIMESTAMP}"

python train.py \
    --stages vision resampler finetune \
    --vision-name "${VISION_MODEL}" \
    --lm-name "${LM_MODEL}" \
    --resampler "${RESAMPLER}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    2>&1 | tee "logs/full_pipeline_${TIMESTAMP}.log"

echo "========================================"
echo "Full training pipeline completed!"
echo "========================================"
echo "Log file: logs/full_pipeline_${TIMESTAMP}.log"
echo "Final model: runs/vlm_finetune/model_final.safetensors"
echo "========================================"