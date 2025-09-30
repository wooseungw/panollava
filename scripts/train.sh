#!/usr/bin/env bash
# Simple training script - relies on config.yaml for all settings
# Updated to use integrated finetune stage (includes instruction tuning)

# Set PYTHONPATH to include src directory
export PYTHONPATH="/data/1_personal/4_SWWOO/panollava/src:$PYTHONPATH"

echo "Starting PanoLLaVA training pipeline..."
echo "Stages: vision → resampler → finetune (with instruction tuning)"
echo ""

# Run training
python scripts/train.py --config configs/default.yaml --stage "vision,resampler,finetune"

# Run evaluation
python scripts/eval.py --config configs/default.yaml --csv-input data/quic360/test.csv "$@"
