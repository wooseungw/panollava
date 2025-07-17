#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

# =============================================================================
# PanoLLaVA Model Comparison Evaluation Script
# =============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "PanoLLaVA Model Comparison Evaluation"
echo "=========================================="

# Configuration
CSV_VAL="${1:-data/quic360/test.csv}"
BATCH_SIZE=4
MAX_NEW_TOKENS=64
TEMPERATURE=0.7
OUTPUT_DIR="comparison_eval_results"

# Model Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen2.5-0.5B"
RESAMPLER="mlp"

# Find the best checkpoints
echo "Searching for model checkpoints..."

# Find finetune checkpoint
FINETUNE_CHECKPOINTS=($(find runs/vlm_finetune/checkpoints -name "*.ckpt" -type f 2>/dev/null | sort))
RESAMPLER_CHECKPOINTS=($(find runs/vlm_resampler/checkpoints -name "*.ckpt" -type f 2>/dev/null | sort))

if [ ${#FINETUNE_CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No finetune checkpoints found in runs/vlm_finetune/checkpoints/"
    echo "Make sure you have completed the finetune stage training"
    exit 1
fi

if [ ${#RESAMPLER_CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No resampler checkpoints found in runs/vlm_resampler/checkpoints/"
    echo "Make sure you have completed the resampler stage training"
    exit 1
fi

# Find best finetune checkpoint
BEST_FINETUNE_CHECKPOINT=""
BEST_FINETUNE_VAL_LOSS="999.999"

for ckpt in "${FINETUNE_CHECKPOINTS[@]}"; do
    if [[ $ckpt =~ val_loss=([0-9]+\.[0-9]+) ]]; then
        VAL_LOSS="${BASH_REMATCH[1]}"
        if (( $(echo "$VAL_LOSS < $BEST_FINETUNE_VAL_LOSS" | bc -l) )); then
            BEST_FINETUNE_VAL_LOSS="$VAL_LOSS"
            BEST_FINETUNE_CHECKPOINT="$ckpt"
        fi
    fi
done

if [ -z "$BEST_FINETUNE_CHECKPOINT" ]; then
    BEST_FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINTS[-1]}"
    echo "Using latest finetune checkpoint: $BEST_FINETUNE_CHECKPOINT"
else
    echo "Using best finetune checkpoint: $BEST_FINETUNE_CHECKPOINT (val_loss: $BEST_FINETUNE_VAL_LOSS)"
fi

# Find best resampler checkpoint
BEST_RESAMPLER_CHECKPOINT=""
BEST_RESAMPLER_VAL_LOSS="999.999"

for ckpt in "${RESAMPLER_CHECKPOINTS[@]}"; do
    if [[ $ckpt =~ val_loss=([0-9]+\.[0-9]+) ]]; then
        VAL_LOSS="${BASH_REMATCH[1]}"
        if (( $(echo "$VAL_LOSS < $BEST_RESAMPLER_VAL_LOSS" | bc -l) )); then
            BEST_RESAMPLER_VAL_LOSS="$VAL_LOSS"
            BEST_RESAMPLER_CHECKPOINT="$ckpt"
        fi
    fi
done

if [ -z "$BEST_RESAMPLER_CHECKPOINT" ]; then
    BEST_RESAMPLER_CHECKPOINT="${RESAMPLER_CHECKPOINTS[-1]}"
    echo "Using latest resampler checkpoint: $BEST_RESAMPLER_CHECKPOINT"
else
    echo "Using best resampler checkpoint: $BEST_RESAMPLER_CHECKPOINT (val_loss: $BEST_RESAMPLER_VAL_LOSS)"
fi

# Validate data file
if [ ! -f "$CSV_VAL" ]; then
    echo "Error: Validation data file not found: $CSV_VAL"
    echo "Usage: $0 [path_to_validation_csv]"
    echo "Example: $0 data/quic360/test.csv"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Finetune model: $BEST_FINETUNE_CHECKPOINT"
echo "  Resampler model: $BEST_RESAMPLER_CHECKPOINT"
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
LOG_FILE="logs/comparison_eval_${TIMESTAMP}.log"

echo "Starting model comparison evaluation... (Log: $LOG_FILE)"

# Run comparison evaluation
python eval_comprehensive.py \
    --stage both \
    --finetune-ckpt "$BEST_FINETUNE_CHECKPOINT" \
    --resampler-ckpt "$BEST_RESAMPLER_CHECKPOINT" \
    --csv-val "$CSV_VAL" \
    --vision-name "$VISION_MODEL" \
    --lm-name "$LM_MODEL" \
    --resampler "$RESAMPLER" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --save-samples 20 \
    --num-workers 0 \
    --do-sample \
    --top-p 0.9 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Model Comparison Evaluation Completed!"
echo "=========================================="
echo "Finetune model: $BEST_FINETUNE_CHECKPOINT"
echo "Resampler model: $BEST_RESAMPLER_CHECKPOINT"
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Show comparison summary if metrics files exist
FINETUNE_METRICS="$OUTPUT_DIR/finetune_metrics_"*.json
RESAMPLER_METRICS="$OUTPUT_DIR/resampler_metrics_"*.json

if ls $FINETUNE_METRICS 1> /dev/null 2>&1 && ls $RESAMPLER_METRICS 1> /dev/null 2>&1; then
    echo ""
    echo "COMPARISON SUMMARY:"
    echo "=================="
    
    if command -v jq &> /dev/null; then
        echo ""
        echo "FINETUNE MODEL METRICS:"
        echo "  BLEU-4:  $(jq -r '.bleu4 // "N/A"' $FINETUNE_METRICS 2>/dev/null)"
        echo "  ROUGE-L: $(jq -r '.rougeL // "N/A"' $FINETUNE_METRICS 2>/dev/null)"
        echo "  METEOR:  $(jq -r '.meteor // "N/A"' $FINETUNE_METRICS 2>/dev/null)"
        
        echo ""
        echo "RESAMPLER MODEL METRICS:"
        echo "  BLEU-4:  $(jq -r '.bleu4 // "N/A"' $RESAMPLER_METRICS 2>/dev/null)"
        echo "  ROUGE-L: $(jq -r '.rougeL // "N/A"' $RESAMPLER_METRICS 2>/dev/null)"
        echo "  METEOR:  $(jq -r '.meteor // "N/A"' $RESAMPLER_METRICS 2>/dev/null)"
        
        # Simple winner calculation (based on BLEU-4)
        FINETUNE_BLEU4=$(jq -r '.bleu4 // 0' $FINETUNE_METRICS 2>/dev/null)
        RESAMPLER_BLEU4=$(jq -r '.bleu4 // 0' $RESAMPLER_METRICS 2>/dev/null)
        
        echo ""
        if (( $(echo "$FINETUNE_BLEU4 > $RESAMPLER_BLEU4" | bc -l) )); then
            echo "🏆 BLEU-4 Winner: FINETUNE MODEL ($FINETUNE_BLEU4 vs $RESAMPLER_BLEU4)"
        elif (( $(echo "$RESAMPLER_BLEU4 > $FINETUNE_BLEU4" | bc -l) )); then
            echo "🏆 BLEU-4 Winner: RESAMPLER MODEL ($RESAMPLER_BLEU4 vs $FINETUNE_BLEU4)"
        else
            echo "🤝 BLEU-4 Tie: Both models scored $FINETUNE_BLEU4"
        fi
        
    else
        echo "Install 'jq' to see detailed comparison: sudo apt-get install jq"
        echo "Finetune metrics: $FINETUNE_METRICS"
        echo "Resampler metrics: $RESAMPLER_METRICS"
    fi
fi

echo ""
echo "Model comparison evaluation complete!"
echo "Check detailed results in: $OUTPUT_DIR"
echo "=========================================="