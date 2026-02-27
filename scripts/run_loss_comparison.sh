#!/usr/bin/env bash
# ============================================================================
# CORA 3-stage loss comparison: VICReg vs Contrastive vs DenseCL
#
# Usage:
#   # Run all 3 experiments sequentially on GPU 0
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_loss_comparison.sh
#
#   # Run a specific experiment only
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_loss_comparison.sh vicreg
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_loss_comparison.sh contrastive
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_loss_comparison.sh densecl
# ============================================================================
set -euo pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
SCRIPT="scripts/train.py"
CONFIG_DIR="configs/cora"
LOG_DIR="runs"

mkdir -p "${LOG_DIR}"

run_experiment() {
    local name="$1"
    local config="${CONFIG_DIR}/${name}.yaml"
    local log="${LOG_DIR}/cora_${name}.log"

    echo "============================================================"
    echo "  CORA 3-stage: ${name}"
    echo "  Config : ${config}"
    echo "  Log    : ${log}"
    echo "  GPU    : ${CUDA_VISIBLE_DEVICES:-auto}"
    echo "  Stages : vision → resampler → finetune"
    echo "============================================================"

    ${PYTHON} ${SCRIPT} --config "${config}" 2>&1 | tee "${log}"

    echo ""
    echo "  ✅ ${name} complete — log: ${log}"
    echo ""
}

# ── Dispatch ──
TARGET="${1:-all}"

case "${TARGET}" in
    vicreg)
        run_experiment vicreg
        ;;
    contrastive)
        run_experiment contrastive
        ;;
    densecl)
        run_experiment densecl
        ;;
    all)
        run_experiment vicreg
        run_experiment contrastive
        run_experiment densecl
        echo "============================================================"
        echo "  All 3 experiments complete."
        echo "  Results:"
        echo "    runs/cora_vicreg/       — VICReg pairwise"
        echo "    runs/cora_contrastive/  — Contrastive (InfoNCE)"
        echo "    runs/cora_densecl/      — DenseCL"
        echo "============================================================"
        ;;
    *)
        echo "Usage: $0 {vicreg|contrastive|densecl|all}"
        exit 1
        ;;
esac
