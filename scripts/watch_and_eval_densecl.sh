#!/bin/bash
# Watch for DenseCL training completion and auto-start evaluation.
# Polls stage_state.json until finetune stage is in completed_stages.
set -euo pipefail

PYTHON=/home/wsw/miniconda3/envs/pano/bin/python
STATE_FILE="runs/cora_densecl/20260218_001/stage_state.json"
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"
OUTPUT_DIR="outputs/cora_densecl/"
LOG_FILE="runs/cora_densecl_eval.log"

echo "[$(date)] Waiting for DenseCL training to complete (watching $STATE_FILE)..."

while true; do
    if [ -f "$STATE_FILE" ]; then
        if python3 -c "
import json, sys
with open('$STATE_FILE') as f:
    state = json.load(f)
if 'finetune' in state.get('completed_stages', []):
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            break
        fi
    fi
    sleep 30
done

CKPT=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
print(state['finetune']['checkpoint'])
")

echo "[$(date)] DenseCL training complete! Checkpoint: $CKPT"
echo "[$(date)] Starting DenseCL evaluation..."

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PYTHON scripts/eval.py \
    --checkpoint "$CKPT" \
    --test-csv "$TEST_CSV" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "[$(date)] DenseCL evaluation complete!"
