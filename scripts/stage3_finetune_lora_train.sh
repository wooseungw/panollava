#!/bin/bash

# =============================================================================
# LoRA를 사용한 Finetune Stage 훈련 스크립트
# =============================================================================

# 설정 파일 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# LoRA 설정 활성화
override_config "use_lora" true
override_config "lora_rank" 16
override_config "lora_alpha" 32
override_config "lora_dropout" 0.1

echo "🚀 Starting LoRA Finetune Stage Training..."
echo "⚙️  Configuration:"
echo "   - Vision Model: $VISION_MODEL"
echo "   - Language Model: $LM_MODEL"
echo "   - Resampler: $RESAMPLER"
echo "   - Crop Strategy: $CROP_STRATEGY"
echo "   - Batch Size: $FINETUNE_BATCH_SIZE"
echo "   - Epochs: $FINETUNE_EPOCHS"
echo "   - Max Text Length: $MAX_TEXT_LENGTH"
echo "   - LoRA Rank: $LORA_RANK"
echo "   - LoRA Alpha: $LORA_ALPHA"
echo "   - LoRA Dropout: $LORA_DROPOUT"
echo "   - Save LoRA Only: $SAVE_LORA_ONLY"
echo ""
echo "💡 Note: LoRA weights will be automatically saved to: ./runs/${CROP_STRATEGY}_finetune_${RESAMPLER}/lora_weights"
echo ""

# 훈련 명령어 구성
TRAIN_CMD="python train.py \
    --csv-train \"$CSV_TRAIN\" \
    --csv-val \"$CSV_VAL\" \
    --vision-name \"$VISION_MODEL\" \
    --lm-name \"$LM_MODEL\" \
    --resampler \"$RESAMPLER\" \
    --stage finetune \
    --crop-strategy \"$CROP_STRATEGY\" \
    --image-size $IMAGE_SIZE \
    --epochs $FINETUNE_EPOCHS \
    --batch-size $FINETUNE_BATCH_SIZE \
    --lr 5e-5 \
    --num-workers $NUM_WORKERS \
    --max-text-length $MAX_TEXT_LENGTH \
    --system-msg \"$FINETUNE_SYSTEM_MSG\" \
    --wandb-project \"$WANDB_PROJECT\" \
    --wandb-name \"${CROP_STRATEGY}_finetune_${RESAMPLER}_lora\" \
    --use-lora \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT"

# LoRA 가중치만 저장할지 결정
if [ "$SAVE_LORA_ONLY" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --save-lora-only"
fi

# 이전 단계 체크포인트 찾기
RESAMPLER_CKPT="./runs/${CROP_STRATEGY}_resampler_${RESAMPLER}/best.ckpt"
if [ -f "$RESAMPLER_CKPT" ]; then
    echo "📂 Found resampler checkpoint: $RESAMPLER_CKPT"
    TRAIN_CMD="$TRAIN_CMD --resume-from \"$RESAMPLER_CKPT\""
else
    echo "⚠️  No resampler checkpoint found at $RESAMPLER_CKPT"
    echo "   Starting finetune stage from scratch"
fi

echo ""
echo "🔧 Executing command:"
echo "$TRAIN_CMD"
echo ""

# 명령어 실행
eval $TRAIN_CMD

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ LoRA Finetune training completed successfully!"
    
    # 결과 파일 확인
    RESULT_DIR="./runs/${CROP_STRATEGY}_finetune_${RESAMPLER}"
    if [ -d "$RESULT_DIR" ]; then
        echo "📁 Results saved in: $RESULT_DIR"
        echo "📋 Available files:"
        ls -la "$RESULT_DIR"/ | head -10
        
        # LoRA 가중치 확인 (항상 저장됨)
        if [ -d "$RESULT_DIR/lora_weights" ]; then
            echo ""
            echo "✅ LoRA weights saved for evaluation:"
            echo "📂 $RESULT_DIR/lora_weights/"
            ls -la "$RESULT_DIR/lora_weights"/ 2>/dev/null || echo "   (Contents may be loading...)"
        else
            echo "⚠️  LoRA weights directory not found"
        fi
        
        # 전체 모델 체크포인트 확인
        if [ -f "$RESULT_DIR/best.ckpt" ]; then
            echo "✅ PyTorch Lightning checkpoint: $RESULT_DIR/best.ckpt"
        fi
        
        if [ -f "$RESULT_DIR/model_final.safetensors" ]; then
            echo "✅ Final model weights: $RESULT_DIR/model_final.safetensors"
        elif [ "$SAVE_LORA_ONLY" = "true" ]; then
            echo "💾 Full model weights skipped (save-lora-only enabled)"
        fi
    fi
    
    echo ""
    echo "🎯 To evaluate the LoRA model:"
    echo "   bash scripts/eval_lora_finetune.sh"
    echo ""
    echo "🔄 To merge LoRA weights back to base model:"
    echo "   python -c \"from panovlm.model import PanoramaVLM; model = PanoramaVLM(...); model.load_lora_weights('$RESULT_DIR/lora_weights'); model.merge_lora_weights()\""
else
    echo ""
    echo "❌ Training failed with exit code $?"
    echo "📋 Check the logs for more details"
    exit 1
fi
