#!/usr/bin/env bash
# ============================================================================
# Run evaluations for CORA experiments after training completes.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_remaining_evals.sh
# ============================================================================
set -euo pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
EVAL="scripts/eval.py"
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"

run_eval() {
    local name="$1"
    local ckpt_pattern="runs/${name}/*/finetune/last.ckpt"
    local output_dir="outputs/${name}"

    # Find the checkpoint
    local ckpt
    ckpt=$(ls -t ${ckpt_pattern} 2>/dev/null | head -1)
    if [[ -z "$ckpt" ]]; then
        echo "⚠️  No finetune checkpoint found for ${name} (pattern: ${ckpt_pattern})"
        echo "   Training may not be complete yet. Skipping."
        return 1
    fi

    echo "============================================================"
    echo "  Evaluating: ${name}"
    echo "  Checkpoint: ${ckpt}"
    echo "  Output:     ${output_dir}"
    echo "============================================================"

    PYTHONUNBUFFERED=1 $PYTHON $EVAL \
        --checkpoint "$ckpt" \
        --test-csv "$TEST_CSV" \
        --output-dir "$output_dir" \
        2>&1 | tee "runs/${name}_eval.log"

    echo "✅ ${name} evaluation complete"
    echo ""
}

echo "=== CORA Remaining Evaluations ==="
echo "Start time: $(date)"
echo ""

# Run evaluations for experiments that have completed training
# (skips any that don't have finetune checkpoints yet)

# B. VICReg (pairwise)
run_eval "cora_vicreg" || true

# D. DenseCL
run_eval "cora_densecl" || true

echo ""
echo "=== ALL REMAINING EVALUATIONS DONE ==="
echo "End time: $(date)"
