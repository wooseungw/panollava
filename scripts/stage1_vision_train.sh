#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
set -e

VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"
CROP_STRATEGY="e2p"
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"
NUM_WORKERS=64
WANDB_PROJECT="panollava-training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs
mkdir -p runs
mkdir -p runs/${CROP_STRATEGY}_vision_${RESAMPLER}

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