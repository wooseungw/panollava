#!/bin/bash
# ==========================================================
# GPU 1 Track B — B1 (Qwen2.5-VL-3B) + B3 (Gemma3-4B) PanoAdapt
# Fixes applied:
#   - panoadapt.py: DenseCLLoss.forward now handles ndim=3 (Qwen stacked tiles)
#   - finetune.py: _compute_densecl else-branch handles 3D hook output (Gemma3)
# ==========================================================
set -o pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

SUCCEEDED=()
FAILED=()

run_task() {
    local label="$1"
    shift
    echo ""
    echo "=== $label ==="
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S KST')"
    if "$@"; then
        echo "  ✅ $label DONE ($(date '+%Y-%m-%d %H:%M:%S KST'))"
        SUCCEEDED+=("$label")
        return 0
    else
        echo "  ❌ $label FAILED (exit=$?, $(date '+%Y-%m-%d %H:%M:%S KST'))"
        FAILED+=("$label")
        return 1
    fi
}

echo "=========================================="
echo "  GPU 1 Track B — B1 + B3 PanoAdapt"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ----------------------------------------------------------
# Smoke Test 1: B1 Qwen2.5-VL-3B PanoAdapt (ndim=3 fix 검증)
# ----------------------------------------------------------
echo ""
echo ">>> [Smoke Test 1] B1 Qwen2.5-VL-3B PanoAdapt (DenseCL ndim=3 fix)"

$PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
cfg.training.num_epochs = 0.001  # ~2 steps
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST B1 PASSED')
" 2>&1 | tee /tmp/smoke_b1_qwen.log

SMOKE_B1=${PIPESTATUS[0]}
if [ $SMOKE_B1 -ne 0 ]; then
    echo "❌ SMOKE TEST B1 FAILED — check /tmp/smoke_b1_qwen.log"
    echo "   Skipping B1 training. Proceeding to B3 smoke test."
    FAILED+=("[B1] Smoke test")
else
    echo "✅ SMOKE TEST B1 PASSED — starting full B1 training"

    # ----------------------------------------------------------
    # B1: Qwen2.5-VL-3B PanoAdapt — full training
    # ----------------------------------------------------------
    run_task "[B1] Qwen2.5-VL-3B PanoAdapt training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml

    if [ "${FAILED[*]}" = "" ] || [[ ! "${FAILED[*]}" =~ "B1.*training" ]]; then
        run_task "[B1] Qwen2.5-VL-3B PanoAdapt eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml \
            --output-dir runs/baseline/panoadapt_qwen25-vl-3b/eval
    fi
fi

# ----------------------------------------------------------
# Smoke Test 2: B3 Gemma3-4B PanoAdapt (3D hook fix 검증)
# ----------------------------------------------------------
echo ""
echo ">>> [Smoke Test 2] B3 Gemma3-4B PanoAdapt (3D hook feature fix)"

$PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('configs/baseline/panoadapt_gemma3_4b.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
cfg.training.num_epochs = 0.001  # ~2 steps
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST B3 PASSED')
" 2>&1 | tee /tmp/smoke_b3_gemma3.log

SMOKE_B3=${PIPESTATUS[0]}
if [ $SMOKE_B3 -ne 0 ]; then
    echo "❌ SMOKE TEST B3 FAILED — check /tmp/smoke_b3_gemma3.log"
    FAILED+=("[B3] Smoke test")
else
    echo "✅ SMOKE TEST B3 PASSED — starting full B3 training"

    # ----------------------------------------------------------
    # B3: Gemma3-4B PanoAdapt — full training
    # ----------------------------------------------------------
    run_task "[B3] Gemma3-4B PanoAdapt training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_gemma3_4b.yaml

    if [ "${FAILED[*]}" = "" ] || [[ ! "${FAILED[*]}" =~ "B3.*training" ]]; then
        run_task "[B3] Gemma3-4B PanoAdapt eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_gemma3_4b.yaml \
            --output-dir runs/baseline/panoadapt_gemma3-4b/eval
    fi
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "=========================================="
echo "  GPU 1 Track B — SUMMARY"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="
echo ""
echo "✅ Succeeded (${#SUCCEEDED[@]}):"
for s in "${SUCCEEDED[@]}"; do echo "   - $s"; done
echo ""
echo "❌ Failed (${#FAILED[@]}):"
for f in "${FAILED[@]}"; do echo "   - $f"; done
echo "=========================================="
