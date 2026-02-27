#!/bin/bash
# Full PanoAdapt training + eval for InternVL3.5-2B on GPU 0
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
CONFIG="configs/baseline/panoadapt_internvl35_2b.yaml"

echo "=========================================="
echo "  PanoAdapt InternVL3.5-2B — Full Training"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Config: $CONFIG"
echo "  Started: $(date)"
echo "=========================================="

# --- Phase 1: Training ---
echo ""
echo "[Phase 1] Training..."
$PYTHON scripts/baseline_finetune.py --config "$CONFIG"
echo "[Phase 1] Training completed: $(date)"

# --- Phase 2: Evaluation ---
echo ""
echo "[Phase 2] Evaluation..."
$PYTHON scripts/baseline_eval.py \
    --config "$CONFIG" \
    --test-csv "runs/baseline/_shared_data/test.csv"
echo "[Phase 2] Evaluation completed: $(date)"

echo ""
echo "=========================================="
echo "  PanoAdapt InternVL3.5-2B — ALL DONE"
echo "  Finished: $(date)"
echo "=========================================="
