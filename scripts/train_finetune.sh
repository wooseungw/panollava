#!/bin/bash
# Stage 3: End-to-End Fine-tuning with LoRA
# ===========================================

set -e  # Exit on error

echo "=== Panorama VLM - Stage 3: Fine-tuning with LoRA ==="
echo "Starting end-to-end fine-tuning with LoRA..."

# 기본 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 이전 stage 체크포인트 자동 탐지
RESAMPLER_CKPT_DIR="./runs/e2p_resampler_mlp"
if [ -f "${RESAMPLER_CKPT_DIR}/best.ckpt" ]; then
    RESUME_FROM="${RESAMPLER_CKPT_DIR}/best.ckpt"
    echo "✓ Found resampler checkpoint: ${RESUME_FROM}"
elif [ -f "${RESAMPLER_CKPT_DIR}/last.ckpt" ]; then
    RESUME_FROM="${RESAMPLER_CKPT_DIR}/last.ckpt"
    echo "✓ Found resampler checkpoint: ${RESUME_FROM}"
else
    echo "⚠️  No resampler checkpoint found in ${RESAMPLER_CKPT_DIR}"
    echo "   Make sure to run resampler training first or specify --resume-from manually"
    RESUME_FROM=""
fi

# LoRA 설정 (환경 변수로 오버라이드 가능)
export PANO_VLM_MODEL_LORA_ENABLED=${USE_LORA:-true}
export PANO_VLM_MODEL_LORA_R=${LORA_R:-16}
export PANO_VLM_MODEL_LORA_ALPHA=${LORA_ALPHA:-32}
export PANO_VLM_MODEL_LORA_DROPOUT=${LORA_DROPOUT:-0.1}

# 학습 설정 (환경 변수로 오버라이드 가능)
# export PANO_VLM_TRAINING_LEARNING_RATE=2e-4
# export PANO_VLM_DATA_BATCH_SIZE=2
# export PANO_VLM_TRAINING_EPOCHS=3

echo "LoRA Configuration:"
echo "  - Enabled: ${PANO_VLM_MODEL_LORA_ENABLED}"
echo "  - Rank: ${PANO_VLM_MODEL_LORA_R}"
echo "  - Alpha: ${PANO_VLM_MODEL_LORA_ALPHA}"
echo "  - Dropout: ${PANO_VLM_MODEL_LORA_DROPOUT}"

# Python 스크립트 실행
python train.py \
    --config-stage finetune \
    --csv-train "${CSV_TRAIN:-data/quic360/train.csv}" \
    --csv-val "${CSV_VAL:-data/quic360/valid.csv}" \
    --wandb-project "${WANDB_PROJECT:-panorama-vlm}" \
    --resume-from "${RESUME_FROM}" \
    $@

echo "✓ Fine-tuning completed!"
echo "Training pipeline finished successfully!"

# LoRA 어댑터 위치 안내
FINETUNE_CKPT_DIR="./runs/e2p_finetune_mlp"
echo ""
echo "LoRA adapter saved in: ${FINETUNE_CKPT_DIR}/"
echo "Use this for inference or further fine-tuning."