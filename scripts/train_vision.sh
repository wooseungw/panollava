#!/bin/bash
# Stage 1: Vision Encoder Training with VICReg Loss
# ==================================================

set -e  # Exit on error

echo "=== Panorama VLM - Stage 1: Vision Training ==="
echo "Starting vision encoder training with VICReg loss..."

# 기본 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 환경 변수를 통한 설정 오버라이드 (선택사항)
# export PANO_VLM_TRAINING_LEARNING_RATE=1e-4
# export PANO_VLM_DATA_BATCH_SIZE=8
# export PANO_VLM_TRAINING_EPOCHS=5

# Python 스크립트 실행
python train.py \
    --config-stage vision \
    --csv-train "${CSV_TRAIN:-data/quic360/train.csv}" \
    --csv-val "${CSV_VAL:-data/quic360/valid.csv}" \
    --wandb-project "${WANDB_PROJECT:-panorama-vlm}" \
    --resume-from "${RESUME_FROM:-}" \
    $@

echo "✓ Vision training completed!"
echo "Next step: run ./scripts/train_resampler.sh"