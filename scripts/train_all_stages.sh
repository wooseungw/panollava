#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
# =============================================================================
# Full 3-Stage Training Pipeline (Separate train.py executions)
# =============================================================================

set -e  # Exit on any error

echo "========================================"
echo "PanoLLaVA Full Training Pipeline"
echo "========================================"

# Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"

CROP_STRATEGY="e2p"  # E2P crop strategy

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

mkdir -p logs
mkdir -p runs
mkdir -p runs/${CROP_STRATEGY}_vision_${RESAMPLER}
mkdir -p runs/${CROP_STRATEGY}_resampler_${RESAMPLER}
mkdir -p runs/${CROP_STRATEGY}_finetune_${RESAMPLER}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
    --epochs 3 \
    --batch-size 16 \
    --resampler "${RESAMPLER}" \
    --crop-strategy "${CROP_STRATEGY}" \
    --csv-train "${CSV_TRAIN}" \
    --csv-val "${CSV_VAL}" \
    --num-workers "${NUM_WORKERS}" \
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
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "finetune_${TIMESTAMP}" \
    --resume-from "${RESAMPLER_CHECKPOINT}" \
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

# Optional: Save final model in different format
echo "Converting final checkpoint to SafeTensors..."
python -c "
import torch
import safetensors.torch as st

# Load checkpoint
ckpt = torch.load('$FINAL_CHECKPOINT', map_location='cpu')
model_state = ckpt['state_dict']

# Remove 'model.' prefix if present
clean_state = {}
for k, v in model_state.items():
    if k.startswith('model.'):
        clean_state[k[6:]] = v
    else:
        clean_state[k] = v

# Save as SafeTensors
st.save_file(clean_state, 'runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/model_final.safetensors')
print('Final model saved as SafeTensors')
"

echo "Final SafeTensors model: runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/model_final.safetensors"