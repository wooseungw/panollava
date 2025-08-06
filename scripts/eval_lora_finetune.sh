#!/bin/bash

# =============================================================================
# LoRA Finetune 모델 평가 스크립트
# =============================================================================

set -e  # Exit on any error

# 설정 파일 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "PanoLLaVA LoRA Finetune Model Evaluation"
echo "=========================================="

# Configuration
CSV_VAL="${1:-data/quic360/test.csv}"
BATCH_SIZE=1
MAX_NEW_TOKENS=64
TEMPERATURE=0.7
OUTPUT_DIR="lora_finetune_eval_results"

echo "Searching for LoRA finetune model checkpoint..."

# LoRA 학습 결과 디렉토리 찾기
FINETUNE_DIR="runs/${CROP_STRATEGY}_finetune_${RESAMPLER}"
LORA_WEIGHTS_DIR="$FINETUNE_DIR/lora_weights"

if [ ! -d "$FINETUNE_DIR" ]; then
    echo "Error: Finetune directory not found: $FINETUNE_DIR"
    echo "Make sure you have completed the finetune stage training"
    exit 1
fi

# 체크포인트 찾기
FINETUNE_CHECKPOINTS=($(find "$FINETUNE_DIR" -name "*.ckpt" -type f 2>/dev/null | sort))

if [ ${#FINETUNE_CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No finetune checkpoints found in $FINETUNE_DIR"
    echo "Make sure you have completed the finetune stage training"
    exit 1
fi

# 최고 체크포인트 선택
BEST_CHECKPOINT=""
BEST_VAL_LOSS="999.999"

for ckpt in "${FINETUNE_CHECKPOINTS[@]}"; do
    if [[ $ckpt =~ val_loss=([0-9]+\.[0-9]+) ]]; then
        VAL_LOSS="${BASH_REMATCH[1]}"
        if (( $(echo "$VAL_LOSS < $BEST_VAL_LOSS" | bc -l) )); then
            BEST_VAL_LOSS="$VAL_LOSS"
            BEST_CHECKPOINT="$ckpt"
        fi
    fi
done

# val_loss 패턴이 없으면 최신 것 사용
if [ -z "$BEST_CHECKPOINT" ]; then
    BEST_CHECKPOINT="${FINETUNE_CHECKPOINTS[-1]}"
    echo "Using latest checkpoint: $BEST_CHECKPOINT"
else
    echo "Using best checkpoint: $BEST_CHECKPOINT (val_loss: $BEST_VAL_LOSS)"
fi

# LoRA 가중치 확인
LORA_PARAM=""
if [ -d "$LORA_WEIGHTS_DIR" ]; then
    echo "✓ LoRA weights found: $LORA_WEIGHTS_DIR"
    LORA_PARAM="--lora-weights-path $LORA_WEIGHTS_DIR"
else
    echo "⚠️  No LoRA weights found. Evaluating base finetune model."
fi

# 검증 데이터 확인
if [ ! -f "$CSV_VAL" ]; then
    echo "Error: Validation data file not found: $CSV_VAL"
    echo "Usage: $0 [path_to_validation_csv]"
    echo "Example: $0 data/quic360/test.csv"
    exit 1
fi

echo "Configuration:"
echo "  Finetune model: $BEST_CHECKPOINT"
if [ -n "$LORA_PARAM" ]; then
    echo "  LoRA weights: $LORA_WEIGHTS_DIR"
fi
echo "  Validation data: $CSV_VAL"
echo "  Batch size: $BATCH_SIZE"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "=========================================="

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# 타임스탬프 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/lora_finetune_eval_${TIMESTAMP}.log"

echo "Starting LoRA finetune model evaluation... (Log: $LOG_FILE)"

# 평가 실행
python eval.py \
    --stage finetune \
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
    $LORA_PARAM \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "LoRA Finetune Model Evaluation Completed!"
echo "=========================================="
echo "Model evaluated: $BEST_CHECKPOINT"
if [ -d "$LORA_WEIGHTS_DIR" ]; then
    echo "LoRA weights: $LORA_WEIGHTS_DIR"
fi
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# 빠른 요약 표시
METRICS_FILE="$OUTPUT_DIR/finetune_metrics_"*.json
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
        
        # LoRA 관련 통계
        echo ""
        echo "MODEL STATISTICS:"
        echo "================="
        echo "Avg prediction length: $(jq -r '.avg_pred_length // "N/A"' $METRICS_FILE 2>/dev/null)"
        echo "Empty predictions: $(jq -r '.empty_predictions // "N/A"' $METRICS_FILE 2>/dev/null)"
    else
        echo "Install 'jq' to see detailed metrics: sudo apt-get install jq"
        echo "Metrics file: $METRICS_FILE"
    fi
fi

echo ""
if [ -d "$LORA_WEIGHTS_DIR" ]; then
    echo "🎉 LoRA finetune model evaluation complete!"
    echo "   Model successfully loaded with LoRA adaptation"
else
    echo "✅ Base finetune model evaluation complete!"
fi
echo "=========================================="
