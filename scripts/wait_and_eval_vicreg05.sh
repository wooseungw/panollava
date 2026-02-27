#!/bin/bash
# Wait for VICReg overlap05 training to finish, then run eval
set -euo pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"
cd /data/1_personal/4_SWWOO/panollava

echo "[$(date)] Waiting for VICReg overlap05 training to complete..."

# Wait until finetune checkpoint appears
while true; do
    CKPT=$(find runs/cora_vicreg_overlap05/ -path "*/finetune/finetune-epoch*.ckpt" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        # Also check stage_state has finetune completed
        STATE=$(find runs/cora_vicreg_overlap05/ -name "stage_state.json" 2>/dev/null | head -1)
        if [ -n "$STATE" ] && grep -q '"finetune"' "$STATE" 2>/dev/null; then
            echo "[$(date)] Training complete! Checkpoint: $CKPT"
            break
        fi
    fi
    sleep 60
done

# Small delay to ensure checkpoint is fully written
sleep 30

echo "[$(date)] Starting VICReg overlap05 greedy eval..."
source /home/wsw/miniconda3/etc/profile.d/conda.sh
conda activate pano
source fix_mamba_cuda.sh 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/eval.py \
    --checkpoint "$CKPT" \
    --test-csv "$TEST_CSV" \
    --output-dir outputs/cora_vicreg_overlap05_greedy \
    2>&1 | tee runs/cora_vicreg_overlap05_eval_greedy.log

echo "[$(date)] VICReg overlap05 eval complete!"
echo "Results: outputs/cora_vicreg_overlap05_greedy/metrics.json"
