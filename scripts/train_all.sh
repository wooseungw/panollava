#!/bin/bash
# Complete 3-Stage Training Pipeline
# ===================================

set -e  # Exit on error

echo "=========================================="
echo "  Panorama VLM - Complete Training Pipeline"
echo "=========================================="
echo ""

# 기본 설정 확인
if [ -z "${CSV_TRAIN}" ] || [ -z "${CSV_VAL}" ]; then
    echo "⚠️  Please set CSV_TRAIN and CSV_VAL environment variables:"
    echo "   export CSV_TRAIN=path/to/train.csv"
    echo "   export CSV_VAL=path/to/val.csv"
    echo ""
    echo "Using default paths..."
    export CSV_TRAIN=${CSV_TRAIN:-"data/quic360/train.csv"}
    export CSV_VAL=${CSV_VAL:-"data/quic360/valid.csv"}
fi

echo "Configuration:"
echo "  - Training data: ${CSV_TRAIN}"
echo "  - Validation data: ${CSV_VAL}"
echo "  - CUDA devices: ${CUDA_VISIBLE_DEVICES:-0}"
echo "  - WandB project: ${WANDB_PROJECT:-panorama-vlm}"
echo ""

# 스크립트 실행 가능하게 만들기
chmod +x scripts/train_*.sh

# Stage 1: Vision Training
echo "🚀 Starting Stage 1: Vision Encoder Training..."
echo "=============================================="
./scripts/train_vision.sh "$@"
echo ""

# Stage 2: Resampler Training  
echo "🚀 Starting Stage 2: Resampler Training..."
echo "=========================================="
./scripts/train_resampler.sh "$@"
echo ""

# Stage 3: Fine-tuning
echo "🚀 Starting Stage 3: Fine-tuning with LoRA..."
echo "============================================="
./scripts/train_finetune.sh "$@"
echo ""

echo "🎉 Complete training pipeline finished successfully!"
echo ""
echo "Results:"
echo "  - Vision model: ./runs/e2p_vision_mlp/"
echo "  - Resampler model: ./runs/e2p_resampler_mlp/"
echo "  - Fine-tuned model: ./runs/e2p_finetune_mlp/"
echo ""
echo "Ready for inference! 🚀"