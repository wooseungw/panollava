#!/bin/bash
# Run native Gemma3-4B training then eval sequentially.
# Usage: bash scripts/run_native_gemma3.sh [GPU_ID]

set -e
GPU=${1:-1}
PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
CONFIG="configs/baseline/native_gemma3_4b.yaml"
TRAIN_LOG="runs/baseline/native_gemma3-4b/train.log"
EVAL_LOG="runs/baseline/native_gemma3-4b/eval_new.log"

cd /data/1_personal/4_SWWOO/panollava

echo "=== [$(date)] Starting Gemma3-4B native training on GPU $GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u scripts/baseline_finetune.py \
    --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

echo "=== [$(date)] Training done. Starting eval on GPU $GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u scripts/baseline_eval.py \
    --config "$CONFIG" 2>&1 | tee "$EVAL_LOG"

echo "=== [$(date)] All done. ==="
