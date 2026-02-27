#!/bin/bash
# ==========================================================
# GPU 0 Track B Queue — PanoAdapt smoke test + B1/B2/B3 training + eval
# ==========================================================
set -o pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
export CUDA_VISIBLE_DEVICES=0

SUCCEEDED=()
FAILED=()

run_task() {
    local label="$1"
    shift
    echo ""
    echo "=== $label ==="
    echo "  Started: $(date)"
    if "$@"; then
        echo "  ✅ $label DONE ($(date))"
        SUCCEEDED+=("$label")
    else
        echo "  ❌ $label FAILED (exit=$?, $(date))"
        FAILED+=("$label")
    fi
    echo ""
}

echo "=========================================="
echo "  GPU 0 Track B Queue"
echo "  Started: $(date)"
echo "=========================================="

# ----------------------------------------------------------
# PanoAdapt smoke test (tiny epoch, ~5 min)
# ----------------------------------------------------------
echo ""
echo ">>> PanoAdapt Smoke Test"
$PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
# Use tiny epoch fraction (~2 steps) for smoke test
cfg.training.num_epochs = 0.001
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST PASSED: PanoAdapt 1-step completed without hang/OOM')
" 2>&1 | tee /tmp/panoadapt_smoke_test.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ SMOKE TEST FAILED — aborting Track B"
    echo "   Check /tmp/panoadapt_smoke_test.log"
    exit 1
fi
echo "✅ SMOKE TEST PASSED — proceeding with Track B"

# ----------------------------------------------------------
# B1: Qwen2.5-VL-3B PanoAdapt (PE + DenseCL)
# ----------------------------------------------------------
run_task "[B1] Qwen2.5-VL-3B PanoAdapt training" \
    $PYTHON scripts/baseline_finetune.py \
    --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml

run_task "[B1] Qwen2.5-VL-3B PanoAdapt eval" \
    $PYTHON scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml \
    --output-dir runs/baseline/panoadapt_qwen25-vl-3b/eval

# ----------------------------------------------------------
# B2: InternVL3.5-2B PanoAdapt (PE + DenseCL)
# ----------------------------------------------------------
run_task "[B2] InternVL3.5-2B PanoAdapt training" \
    $PYTHON scripts/baseline_finetune.py \
    --config configs/baseline/panoadapt_internvl35_2b.yaml

run_task "[B2] InternVL3.5-2B PanoAdapt eval" \
    $PYTHON scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_internvl35_2b.yaml \
    --output-dir runs/baseline/panoadapt_internvl35-2b/eval

# ----------------------------------------------------------
# B3: Gemma3-4B PanoAdapt (PE + DenseCL)
# ----------------------------------------------------------
run_task "[B3] Gemma3-4B PanoAdapt training" \
    $PYTHON scripts/baseline_finetune.py \
    --config configs/baseline/panoadapt_gemma3_4b.yaml

run_task "[B3] Gemma3-4B PanoAdapt eval" \
    $PYTHON scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_gemma3_4b.yaml \
    --output-dir runs/baseline/panoadapt_gemma3-4b/eval

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "=========================================="
echo "  Track B — SUMMARY"
echo "  Finished: $(date)"
echo "=========================================="
echo ""
echo "✅ Succeeded (${#SUCCEEDED[@]}):"
for s in "${SUCCEEDED[@]}"; do echo "   - $s"; done
echo ""
echo "❌ Failed (${#FAILED[@]}):"
for f in "${FAILED[@]}"; do echo "   - $f"; done
echo "=========================================="