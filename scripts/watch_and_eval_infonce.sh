#!/bin/bash
# Watch for VICReg evaluations to finish on GPU 0, then run InfoNCE eval.
# Detects completion by checking if eval.py processes are still running.
set -euo pipefail

PYTHON=/home/wsw/miniconda3/envs/pano/bin/python
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"
CKPT="runs/cora_contrastive/20260216_001/finetune/last.ckpt"
OUTPUT_DIR="outputs/cora_contrastive/"
LOG_FILE="runs/cora_contrastive_eval_v3.log"

echo "[$(date)] Waiting for VICReg evals to finish on GPU 0..."
echo "[$(date)] Will run InfoNCE eval once GPU 0 is free."

while true; do
    if ! pgrep -f "scripts/eval.py.*cora_vicreg" > /dev/null 2>&1; then
        break
    fi
    sleep 60
done

echo "[$(date)] GPU 0 is free. Starting InfoNCE eval..."
echo "[$(date)] Checkpoint: $CKPT"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PYTHON scripts/eval.py \
    --checkpoint "$CKPT" \
    --test-csv "$TEST_CSV" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "[$(date)] InfoNCE evaluation complete!"
