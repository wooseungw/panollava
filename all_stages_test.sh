#!/bin/bash

# Panorama-VLM 3-stage 학습 스크립트 예시
# 각 스테이지별 하이퍼파라미터는 필요에 따라 수정 가능

CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"
VISION_NAME="google/siglip-base-patch16-224"
LM_NAME="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"
PROJECT="panorama-vlm"

# 1. VICReg(vision) stage
python train.py \
  --stage vision \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 64 \
  --vicreg-loss-weight 1.0 \
  --csv-train "$CSV_TRAIN" \
  --csv-val "$CSV_VAL" \
  --vision-name "$VISION_NAME" \
  --lm-name "$LM_NAME" \
  --resampler "$RESAMPLER" \
  --wandb-project "$PROJECT" \
  --max-txt-len 32 \
  --num-workers 16

# 2. Resampler stage (사전학습)
RESAMP_CKPT=$(ls -t ./runs/vlm_vision/checkpoints/*.ckpt 2>/dev/null | head -n 1)
if [ -z "$RESAMP_CKPT" ]; then
  echo "[WARN] No checkpoint found for resampler stage."
  RESAMP_CKPT=""
else
  echo "[INFO] Using checkpoint for resampler: $RESAMP_CKPT"
fi
python train.py \
  --stage resampler \
  --epochs 1 \
  --lr 2e-6 \
  --batch-size 4 \
  --vicreg-loss-weight 0.0 \
  --csv-train "$CSV_TRAIN" \
  --csv-val "$CSV_VAL" \
  --vision-name "$VISION_NAME" \
  --lm-name "$LM_NAME" \
  --resampler "$RESAMPLER" \
  --wandb-project "$PROJECT" \
  --max-txt-len 128 \
  --num-workers 16 \
  ${RESAMP_CKPT:+--resume-from "$RESAMP_CKPT"}

FINETUNE_CKPT=$(ls -t ./runs/vlm_resampler/checkpoints/*.ckpt 2>/dev/null | head -n 1)
if [ -z "$FINETUNE_CKPT" ]; then
  echo "[WARN] No checkpoint found for finetune stage."
  FINETUNE_CKPT=""
else
  echo "[INFO] Using checkpoint for finetune: $FINETUNE_CKPT"
fi
python train.py \
  --stage finetune \
  --epochs 1 \
  --lr 2e-6 \
  --batch-size 4 \
  --vicreg-loss-weight 0.0 \
  --csv-train "$CSV_TRAIN" \
  --csv-val "$CSV_VAL" \
  --vision-name "$VISION_NAME" \
  --lm-name "$LM_NAME" \
  --resampler "$RESAMPLER" \
  --wandb-project "$PROJECT" \
  --max-txt-len 128 \
  --num-workers 16 \
  ${FINETUNE_CKPT:+--resume-from "$FINETUNE_CKPT"}

# 실행 전, 경로/하이퍼파라미터를 프로젝트에 맞게 수정하세요.
