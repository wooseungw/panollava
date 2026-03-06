#!/usr/bin/env bash
# Run all 4 multi-view baseline evaluations sequentially on GPU 1
set -e

PYTHON=/home/wsw/miniconda3/envs/pano/bin/python
TEST_CSV=/data/1_personal/4_SWWOO/panollava/runs/baseline/_shared_data/test.csv
SCRIPT=/data/1_personal/4_SWWOO/panollava/scripts/baseline_eval.py
WORKDIR=/data/1_personal/4_SWWOO/panollava

export CUDA_VISIBLE_DEVICES=1
cd "$WORKDIR"

EXPERIMENTS=(
    "dynamic_qwen25_3b"
    "anyres_e2p_qwen25_3b"
    "cubemap_qwen25_3b"
    "pinhole_qwen25_3b"
)

for exp in "${EXPERIMENTS[@]}"; do
    CONFIG="configs/baseline/${exp}.yaml"
    echo ""
    echo "============================================================"
    echo "STARTING EVAL: $exp"
    echo "Config: $CONFIG"
    echo "Time: $(date)"
    echo "============================================================"
    
    $PYTHON "$SCRIPT" --config "$CONFIG" --test-csv "$TEST_CSV" 2>&1
    
    echo ""
    echo "✅ FINISHED EVAL: $exp at $(date)"
done

echo ""
echo "============================================================"
echo "=== ALL 4 MULTI-VIEW EVALUATIONS COMPLETE ==="
echo "============================================================"
echo ""

# Print summary
for exp_dir in dynamic_qwen25-vl-3b anyres_e2p_qwen25-vl-3b cubemap_qwen25-vl-3b pinhole_qwen25-vl-3b; do
    METRICS_FILE="runs/baseline/${exp_dir}/qwen25-vl-3b/eval/metrics.json"
    echo "--- $exp_dir ---"
    if [ -f "$METRICS_FILE" ]; then
        cat "$METRICS_FILE"
    else
        echo "  metrics.json NOT FOUND at $METRICS_FILE"
    fi
    echo ""
done
