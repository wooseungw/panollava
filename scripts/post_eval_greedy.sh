#!/bin/bash
# Post-processing after ALL greedy evals complete:
# 1. Collect results into comparison tables
# 2. Run LLM-as-judge on best CORA variant (DenseCL)
set -euo pipefail

PYTHON=/home/wsw/miniconda3/envs/pano/bin/python

echo "[$(date)] Waiting for ALL greedy evaluations to complete..."

# Wait until all 4 greedy output dirs have metrics.json
DIRS=(
    "outputs/cora_densecl_greedy"
    "outputs/cora_vicreg_batchwise_greedy"
    "outputs/cora_vicreg_greedy"
    "outputs/cora_contrastive_greedy"
)

while true; do
    all_done=true
    for dir in "${DIRS[@]}"; do
        if [ ! -f "$dir/metrics.json" ]; then
            all_done=false
            break
        fi
    done
    if $all_done; then
        break
    fi
    sleep 60
done

echo "[$(date)] All greedy evaluations complete!"

# 1. Collect results
echo "[$(date)] Running collect_results.py..."
"$PYTHON" scripts/collect_results.py
"$PYTHON" scripts/collect_results.py --latex
"$PYTHON" scripts/collect_results.py --csv outputs/all_results_greedy.csv

echo "[$(date)] Results collected!"

# 2. LLM-as-judge on DenseCL (best variant)
echo "[$(date)] Running LLM-as-judge on DenseCL (greedy)..."
source /data/1_personal/4_SWWOO/panollava/.env

"$PYTHON" scripts/eval.py \
    --csv-input outputs/cora_densecl_greedy/predictions.csv \
    --output-dir outputs/cora_densecl_greedy/ \
    --skip-metrics \
    --llm-judge \
    --judge-model gpt-4.1-mini \
    2>&1 | tee runs/cora_densecl_llm_judge_greedy.log

echo "[$(date)] All post-processing complete!"
