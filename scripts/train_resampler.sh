#!/bin/bash
# Stage 2: Resampler Training
# ===========================

set -e  # Exit on error

echo "=== Panorama VLM - Stage 2: Resampler Training ==="
echo "Starting resampler training with vision encoder..."

# 기본 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 이전 stage 체크포인트 자동 탐지
VISION_CKPT_DIR="./runs/e2p_vision_mlp"
if [ -f "${VISION_CKPT_DIR}/best.ckpt" ]; then
    RESUME_FROM="${VISION_CKPT_DIR}/best.ckpt"
    echo "✓ Found vision checkpoint: ${RESUME_FROM}"
elif [ -f "${VISION_CKPT_DIR}/last.ckpt" ]; then
    RESUME_FROM="${VISION_CKPT_DIR}/last.ckpt"
    echo "✓ Found vision checkpoint: ${RESUME_FROM}"
else
    echo "⚠️  No vision checkpoint found in ${VISION_CKPT_DIR}"
    echo "   Make sure to run vision training first or specify --resume-from manually"
    RESUME_FROM=""
fi

# 환경 변수를 통한 설정 오버라이드 (선택사항)
# export PANO_VLM_TRAINING_LEARNING_RATE=5e-5
# export PANO_VLM_DATA_BATCH_SIZE=6
# export PANO_VLM_VICREG_LOSS_WEIGHT=0.5

# Python 스크립트 실행
python train.py \
    --config-stage resampler \
    --csv-train "${CSV_TRAIN:-data/quic360/train.csv}" \
    --csv-val "${CSV_VAL:-data/quic360/valid.csv}" \
    --wandb-project "${WANDB_PROJECT:-panorama-vlm}" \
    --resume-from "${RESUME_FROM}" \
    $@

echo "✓ Resampler training completed!"
echo "Next step: run ./scripts/train_finetune.sh"