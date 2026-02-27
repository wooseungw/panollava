#!/bin/bash
# ==========================================================
# GPU 0 Phase 2 Queue — PanoAdapt smoke test + Track A evals + Track B training/eval
# Runs AFTER A3+A6+A7 training queue completes
# ==========================================================
set -o pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
TEST_CSV="/data/1_personal/4_SWWOO/panollava/runs/baseline/_shared_data/test.csv"
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
echo "  GPU 0 Phase 2 Queue"
echo "  Started: $(date)"
echo "=========================================="

# ----------------------------------------------------------
# Phase 2-0: PanoAdapt smoke test (1 step, ~5 min)
# ----------------------------------------------------------
echo ""
echo ">>> Phase 2-0: PanoAdapt Smoke Test"
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
    echo "❌ SMOKE TEST FAILED — aborting Track B training"
    echo "   Check /tmp/panoadapt_smoke_test.log"
    echo "   Track A evals will still run."
    SMOKE_OK=false
else
    echo "✅ SMOKE TEST PASSED"
    SMOKE_OK=true
fi

# ----------------------------------------------------------
# Phase 2-1: Track A evals (A2, A3, A6, A7)
# ----------------------------------------------------------
echo ""
echo ">>> Phase 2-1: Track A Evals"

# A2: InternVL3.5-2B (adapter exists, was missed from eval-queue)
run_task "[A2] InternVL3.5-2B eval" \
    $PYTHON scripts/baseline_eval.py \
    --config configs/baseline/native_internvl35_2b.yaml \
    --output-dir runs/baseline/native_internvl35-2b/eval

# A3: InternVL3.5-1B (just trained)
run_task "[A3] InternVL3.5-1B eval" \
    $PYTHON scripts/baseline_eval.py \
    --config configs/baseline/native_internvl35_1b.yaml \
    --output-dir runs/baseline/native_internvl35-1b/eval

# A6: InternVL2.5-4B — SKIPPED (InternVL2.5 custom forward() incompatible with HF Trainer)
echo "  [A6] InternVL2.5-4B SKIPPED — architecture incompatible"

# A7: InternVL2.5-2B — SKIPPED (same reason)
echo "  [A7] InternVL2.5-2B SKIPPED — architecture incompatible"

# ----------------------------------------------------------
# Phase 2-2: Track B training + eval (B1 → B2 → B3)
# ----------------------------------------------------------
if [ "$SMOKE_OK" = true ]; then
    echo ""
    echo ">>> Phase 2-2: Track B PanoAdapt Training + Eval"

    # B1: Qwen2.5-VL-3B PanoAdapt (PE + DenseCL)
    run_task "[B1] Qwen2.5-VL-3B PanoAdapt training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml

    run_task "[B1] Qwen2.5-VL-3B PanoAdapt eval" \
        $PYTHON scripts/baseline_eval.py \
        --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml \
        --output-dir runs/baseline/panoadapt_qwen25-vl-3b/eval

    # B2: InternVL3.5-2B PanoAdapt (PE + DenseCL)
    run_task "[B2] InternVL3.5-2B PanoAdapt training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_internvl35_2b.yaml

    run_task "[B2] InternVL3.5-2B PanoAdapt eval" \
        $PYTHON scripts/baseline_eval.py \
        --config configs/baseline/panoadapt_internvl35_2b.yaml \
        --output-dir runs/baseline/panoadapt_internvl35-2b/eval

    # B3: Gemma3-4B PanoAdapt (PE + DenseCL)
    run_task "[B3] Gemma3-4B PanoAdapt training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_gemma3_4b.yaml

    run_task "[B3] Gemma3-4B PanoAdapt eval" \
        $PYTHON scripts/baseline_eval.py \
        --config configs/baseline/panoadapt_gemma3_4b.yaml \
        --output-dir runs/baseline/panoadapt_gemma3-4b/eval
else
    echo ""
    echo "⚠️  Track B SKIPPED — smoke test failed"
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "=========================================="
echo "  GPU 0 Phase 2 — SUMMARY"
echo "  Finished: $(date)"
echo "=========================================="
echo ""
echo "✅ Succeeded (${#SUCCEEDED[@]}):"
for s in "${SUCCEEDED[@]}"; do echo "   - $s"; done
echo ""
echo "❌ Failed (${#FAILED[@]}):"
for f in "${FAILED[@]}"; do echo "   - $f"; done
echo ""
if [ "$SMOKE_OK" != true ]; then
    echo "⚠️  Track B was SKIPPED due to smoke test failure"
fi
echo "=========================================="
