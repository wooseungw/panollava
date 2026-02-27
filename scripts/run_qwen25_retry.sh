#!/bin/bash
# ==========================================================
# Qwen2.5-VL-3B Retry Queue — GPU 0
#
# Re-runs experiments that previously failed:
#   - B1 (DenseCL): smoke test crashed with old ndim=3 bug (now fixed)
#   - VICReg-pw 50%: OOM on GPU 1 while InternVL was occupying memory
#
# Fix applied: all configs now include attn.qkv + attn.proj in
# lora.target_modules → 0 → 64 vision LoRA modules
# ==========================================================
set -o pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
export CUDA_VISIBLE_DEVICES=0
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

smoke_test() {
    local label="$1"
    local config="$2"
    local logfile="$3"
    echo ""
    echo ">>> [Smoke Test] $label"
    $PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('$config') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
cfg.training.num_epochs = 0.001
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST PASSED')
" 2>&1 | tee "$logfile"
    return ${PIPESTATUS[0]}
}

echo "=========================================="
echo "  Qwen2.5-VL-3B Retry Queue — GPU 0"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ----------------------------------------------------------
# [1/2] B1: PE + DenseCL (vision LoRA fix 적용)
# ----------------------------------------------------------
smoke_test "B1 Qwen2.5 DenseCL" \
    configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml \
    /tmp/smoke_qwen25_b1_retry.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ SMOKE TEST FAILED — check /tmp/smoke_qwen25_b1_retry.log"
    FAILED+=("[B1] Smoke test")
else
    echo "  ✅ SMOKE TEST PASSED"

    rm -rf runs/baseline/panoadapt_qwen25-vl-3b/checkpoints
    rm -rf runs/baseline/panoadapt_qwen25-vl-3b/lora_adapter

    run_task "[B1] Qwen2.5-VL-3B DenseCL training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml

    if [[ ! "${FAILED[*]}" =~ "B1.*training" ]]; then
        run_task "[B1] Qwen2.5-VL-3B DenseCL eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml \
            --output-dir runs/baseline/panoadapt_qwen25-vl-3b/eval
    fi
fi

# ----------------------------------------------------------
# [2/2] VICReg-pw 50%: PE + VICReg-pairwise (vision LoRA fix 적용)
# ----------------------------------------------------------
smoke_test "VICReg-pw Qwen2.5" \
    configs/baseline/panoadapt_vicreg_pairwise_qwen25_3b.yaml \
    /tmp/smoke_qwen25_vicreg_retry.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ SMOKE TEST FAILED — check /tmp/smoke_qwen25_vicreg_retry.log"
    FAILED+=("[VICReg-pw] Smoke test")
else
    echo "  ✅ SMOKE TEST PASSED"

    rm -rf runs/baseline/panoadapt_vicreg_pairwise_qwen25-vl-3b/checkpoints
    rm -rf runs/baseline/panoadapt_vicreg_pairwise_qwen25-vl-3b/lora_adapter

    run_task "[VICReg-pw] Qwen2.5-VL-3B training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_vicreg_pairwise_qwen25_3b.yaml

    if [[ ! "${FAILED[*]}" =~ "VICReg-pw.*training" ]]; then
        run_task "[VICReg-pw] Qwen2.5-VL-3B eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_vicreg_pairwise_qwen25_3b.yaml \
            --output-dir runs/baseline/panoadapt_vicreg_pairwise_qwen25-vl-3b/eval
    fi
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "=========================================="
echo "  Qwen2.5-VL-3B Retry — SUMMARY"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="
echo ""
echo "✅ Succeeded (${#SUCCEEDED[@]}):"
for s in "${SUCCEEDED[@]}"; do echo "   - $s"; done
echo ""
echo "❌ Failed (${#FAILED[@]}):"
for f in "${FAILED[@]}"; do echo "   - $f"; done
echo "=========================================="
