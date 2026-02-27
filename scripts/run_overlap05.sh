#!/bin/bash
# Run overlap 0.5 experiments: VICReg retrain + eval both DenseCL & VICReg
set -euo pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"
ROOT="/data/1_personal/4_SWWOO/panollava"
cd "$ROOT"

source /home/wsw/miniconda3/etc/profile.d/conda.sh
source fix_mamba_cuda.sh 2>/dev/null || true
conda activate pano

echo "============================================"
echo "  Step 1/3: VICReg overlap05 — resume train"
echo "============================================"
$PYTHON scripts/train.py \
    --config configs/cora/vicreg_overlap05.yaml \
    --resume auto \
    2>&1 | tee runs/vicreg_overlap05_resume.log

echo ""
echo "============================================"
echo "  Step 2/3: DenseCL overlap05 — greedy eval"
echo "============================================"
CKPT_DENSECL="runs/cora_densecl_overlap05/20260219_001/finetune/finetune-epoch=00-val_loss=2.5935.ckpt"
$PYTHON scripts/eval.py \
    --checkpoint "$CKPT_DENSECL" \
    --test-csv "$TEST_CSV" \
    --output-dir outputs/cora_densecl_overlap05_greedy \
    2>&1 | tee runs/cora_densecl_overlap05_eval_greedy.log

echo ""
echo "============================================"
echo "  Step 3/3: VICReg overlap05 — greedy eval"
echo "============================================"
# Find the best finetune checkpoint (prefer named over last.ckpt)
CKPT_VICREG=$(find runs/cora_vicreg_overlap05/ -path "*/finetune/finetune-epoch*.ckpt" 2>/dev/null | head -1)
if [ -z "$CKPT_VICREG" ]; then
    CKPT_VICREG=$(find runs/cora_vicreg_overlap05/ -path "*/finetune/last.ckpt" 2>/dev/null | head -1)
fi
if [ -z "$CKPT_VICREG" ]; then
    echo "ERROR: No VICReg overlap05 finetune checkpoint found!"
    exit 1
fi
echo "Using checkpoint: $CKPT_VICREG"
$PYTHON scripts/eval.py \
    --checkpoint "$CKPT_VICREG" \
    --test-csv "$TEST_CSV" \
    --output-dir outputs/cora_vicreg_overlap05_greedy \
    2>&1 | tee runs/cora_vicreg_overlap05_eval_greedy.log

echo ""
echo "============================================"
echo "  All done! Results:"
echo "============================================"
echo "DenseCL overlap05: outputs/cora_densecl_overlap05_greedy/metrics.json"
echo "VICReg  overlap05: outputs/cora_vicreg_overlap05_greedy/metrics.json"
