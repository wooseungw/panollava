#!/bin/bash
# Monitor PanoAdapt VICReg 25% training and auto-trigger evaluation on completion

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
PANOADAPT_SESSION="panoadapt-vicreg-25"
TEST_CSV="/data/1_personal/4_SWWOO/panollava/runs/baseline/_shared_data/test.csv"

echo "🔍 Monitoring panoadapt-vicreg-25 (25% overlap) training..."

# Function to check if training is complete
check_training_complete() {
    tmux capture-pane -t $PANOADAPT_SESSION -p | grep -q "Saving model checkpoint" && echo "complete" || echo "running"
}

# Wait for training to complete
while [ "$(check_training_complete)" = "running" ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training in progress..."
    sleep 60
done

echo "✅ Training complete! Waiting 30 seconds before evaluation..."
sleep 30

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find runs/baseline/panoadapt_vicreg_pairwise_internvl35-2b_25overlap -name "final" -type d | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found!"
    exit 1
fi

echo "📊 Starting evaluation with checkpoint: $LATEST_CHECKPOINT"
echo "Test CSV: $TEST_CSV"

# Run evaluation
$PYTHON -u scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b_25overlap.yaml \
    --test-csv "$TEST_CSV" \
    --output-dir "outputs/panoadapt_vicreg_pairwise_internvl35-2b_25overlap_greedy" \
    2>&1 | tee "outputs/panoadapt_vicreg_pairwise_internvl35-2b_25overlap_greedy/eval.log"

echo "✅ Evaluation complete!"
echo "Results saved to: outputs/panoadapt_vicreg_pairwise_internvl35-2b_25overlap_greedy/metrics.json"
