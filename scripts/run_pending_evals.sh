#!/bin/bash
# Run all pending evals sequentially on GPU 1
# Track A: native VLM evals, Track C: CORA ablation evals
set -e

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
TEST_CSV="runs/baseline/_shared_data/test.csv"
export CUDA_VISIBLE_DEVICES=1

echo "=============================="
echo "Starting eval queue on GPU 1"
echo "=============================="

# --- Track A: Native VLM evals ---
echo ""
echo "=== [1/5] Track A: Qwen2.5-VL-3B native eval ==="
$PYTHON scripts/baseline_eval.py \
    --config configs/baseline/native_qwen25_3b.yaml \
    --output-dir runs/baseline/native_qwen25-vl-3b/eval
echo "=== [1/5] DONE ==="

echo ""
echo "=== [2/5] Track A: Gemma3-4B native eval ==="
$PYTHON scripts/baseline_eval.py \
    --config configs/baseline/native_gemma3_4b.yaml \
    --output-dir runs/baseline/native_gemma3-4b/eval
echo "=== [2/5] DONE ==="

# --- Track C: CORA ablation evals ---
echo ""
echo "=== [3/5] Track C: CORA resize eval ==="
$PYTHON scripts/eval.py \
    --checkpoint runs/cora_ablation_resize/20260222_001/finetune/finetune-epoch=00-val_loss=2.6596.ckpt \
    --test-csv $TEST_CSV \
    --output-dir outputs/cora_ablation_resize_greedy
echo "=== [3/5] DONE ==="

echo ""
echo "=== [4/5] Track C: CORA cubemap eval ==="
$PYTHON scripts/eval.py \
    --checkpoint runs/cora_ablation_cubemap/20260222_001/finetune/finetune-epoch=00-val_loss=2.6310.ckpt \
    --test-csv $TEST_CSV \
    --output-dir outputs/cora_ablation_cubemap_greedy
echo "=== [4/5] DONE ==="

echo ""
echo "=== [5/5] Track C: CORA e2p eval ==="
$PYTHON scripts/eval.py \
    --checkpoint runs/cora_ablation_e2p/20260222_001/finetune/finetune-epoch=00-val_loss=2.6254.ckpt \
    --test-csv $TEST_CSV \
    --output-dir outputs/cora_ablation_e2p_greedy
echo "=== [5/5] DONE ==="

echo ""
echo "=============================="
echo "All 5 evals completed!"
echo "=============================="
