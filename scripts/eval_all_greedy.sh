#!/bin/bash
# Re-evaluate ALL CORA variants with greedy decoding (do_sample=False).
# 
# Previously all CORA evals used Qwen3's default stochastic sampling
# (do_sample=True, temperature=0.6) while baselines used greedy decoding.
# This script fixes the comparison by re-running with the corrected generator.
#
# Usage:
#   GPU 0: DenseCL + VICReg-Batch
#   GPU 1: VICReg-Pair + InfoNCE
#
#   bash scripts/eval_all_greedy.sh gpu0   # Run DenseCL + VICReg-Batch on GPU 0
#   bash scripts/eval_all_greedy.sh gpu1   # Run VICReg-Pair + InfoNCE on GPU 1
set -euo pipefail

PYTHON=/home/wsw/miniconda3/envs/pano/bin/python
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"

run_eval() {
    local name="$1"
    local ckpt="$2"
    local outdir="$3"
    local gpu="$4"
    local logfile="$5"

    echo "[$(date)] Starting $name eval on GPU $gpu → $outdir"
    echo "[$(date)] Checkpoint: $ckpt"

    # Remove stale partial checkpoints from previous (stochastic) runs
    rm -f "$outdir/predictions_partial.csv"

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" scripts/eval.py \
        --checkpoint "$ckpt" \
        --test-csv "$TEST_CSV" \
        --output-dir "$outdir" \
        2>&1 | tee "$logfile"

    echo "[$(date)] $name evaluation complete!"
    echo ""
}

case "${1:-help}" in
    gpu0)
        # GPU 0: DenseCL + VICReg-Batch
        run_eval "DenseCL" \
            "runs/cora_densecl/20260218_001/finetune/finetune-epoch=00-val_loss=2.5932.ckpt" \
            "outputs/cora_densecl_greedy" \
            "0" \
            "runs/cora_densecl_eval_greedy.log"

        run_eval "VICReg-Batch" \
            "runs/cora_vicreg_batchwise/20260217_001/finetune/last.ckpt" \
            "outputs/cora_vicreg_batchwise_greedy" \
            "0" \
            "runs/cora_vicreg_batchwise_eval_greedy.log"
        ;;

    gpu1)
        # GPU 1: VICReg-Pair + InfoNCE
        run_eval "VICReg-Pair" \
            "runs/cora_vicreg/20260218_001/finetune/last.ckpt" \
            "outputs/cora_vicreg_greedy" \
            "1" \
            "runs/cora_vicreg_eval_greedy.log"

        run_eval "InfoNCE" \
            "runs/cora_contrastive/20260216_001/finetune/last.ckpt" \
            "outputs/cora_contrastive_greedy" \
            "1" \
            "runs/cora_contrastive_eval_greedy.log"
        ;;

    *)
        echo "Usage: $0 {gpu0|gpu1}"
        echo "  gpu0: DenseCL + VICReg-Batch on GPU 0"
        echo "  gpu1: VICReg-Pair + InfoNCE on GPU 1"
        exit 1
        ;;
esac

echo "[$(date)] All evaluations for this GPU complete!"
