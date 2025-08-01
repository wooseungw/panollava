#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

# =============================================================================
# PanoLLaVA Resampler Model Evaluation Script
# =============================================================================

set -e  # Exit on any error
# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
echo "=========================================="
echo "PanoLLaVA Resampler Model Evaluation"
echo "=========================================="

# Configuration
CSV_VAL="${1:-data/quic360/test.csv}"
BATCH_SIZE=4
MAX_NEW_TOKENS=64
TEMPERATURE=0.7
OUTPUT_DIR="resampler_eval_results"


echo "Searching for resampler model checkpoint..."

RESAMPLER_CHECKPOINTS=($(find runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/resampler -name "*.ckpt" -type f 2>/dev/null | sort))

if [ ${#RESAMPLER_CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No resampler checkpoints found in runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/resampler/"
    echo "Make sure you have completed the resampler stage training"
    exit 1
fi

# Use the best checkpoint (lowest validation loss)
BEST_CHECKPOINT=""
BEST_VAL_LOSS="999.999"

for ckpt in "${RESAMPLER_CHECKPOINTS[@]}"; do
    if [[ $ckpt =~ val_loss=([0-9]+\.[0-9]+) ]]; then
        VAL_LOSS="${BASH_REMATCH[1]}"
        if (( $(echo "$VAL_LOSS < $BEST_VAL_LOSS" | bc -l) )); then
            BEST_VAL_LOSS="$VAL_LOSS"
            BEST_CHECKPOINT="$ckpt"
        fi
    fi
done

# If no checkpoint with val_loss pattern found, use the latest one
if [ -z "$BEST_CHECKPOINT" ]; then
    BEST_CHECKPOINT="${RESAMPLER_CHECKPOINTS[-1]}"
    echo "Using latest checkpoint: $BEST_CHECKPOINT"
else
    echo "Using best checkpoint: $BEST_CHECKPOINT (val_loss: $BEST_VAL_LOSS)"
fi

# Validate data file
if [ ! -f "$CSV_VAL" ]; then
    echo "Error: Validation data file not found: $CSV_VAL"
    echo "Usage: $0 [path_to_validation_csv]"
    echo "Example: $0 data/quic360/test.csv"
    exit 1
fi

echo "Configuration:"
echo "  Resampler model: $BEST_CHECKPOINT"
echo "  Validation data: $CSV_VAL"
echo "  Batch size: $BATCH_SIZE"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Generate timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/resampler_eval_${TIMESTAMP}.log"

echo "Starting resampler model evaluation... (Log: $LOG_FILE)"

# Run evaluation
python eval_comprehensive.py \
    --stage resampler \
    --ckpt "$BEST_CHECKPOINT" \
    --csv-val "$CSV_VAL" \
    --vision-name "$VISION_MODEL" \
    --lm-name "$LM_MODEL" \
    --resampler "$RESAMPLER" \
    --crop-strategy "$CROP_STRATEGY" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --save-samples 20 \
    --num-workers 0 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Resampler Model Evaluation Completed!"
echo "=========================================="
echo "Model evaluated: $BEST_CHECKPOINT"
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Show quick summary if metrics file exists
METRICS_FILE="$OUTPUT_DIR/resampler_metrics_"*.json
if ls $METRICS_FILE 1> /dev/null 2>&1; then
    echo ""
    echo "EVALUATION METRICS:"
    echo "=================="
    
    if command -v jq &> /dev/null; then
        echo "BLEU-4:  $(jq -r '.bleu4 // "N/A"' $METRICS_FILE 2>/dev/null)"
        echo "ROUGE-L: $(jq -r '.rougeL // "N/A"' $METRICS_FILE 2>/dev/null)"
        echo "METEOR:  $(jq -r '.meteor // "N/A"' $METRICS_FILE 2>/dev/null)"
        
        # CIDEr if available
        CIDER=$(jq -r '.cider // empty' $METRICS_FILE 2>/dev/null)
        if [ -n "$CIDER" ] && [ "$CIDER" != "null" ]; then
            echo "CIDEr:   $CIDER"
        fi
    else
        echo "Install 'jq' to see detailed metrics: sudo apt-get install jq"
        echo "Metrics file: $METRICS_FILE"
    fi
fi

echo ""
echo "Resampler model evaluation complete!"
echo "=========================================="