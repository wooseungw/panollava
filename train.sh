#!/usr/bin/env bash
# ===============================================================
# run_train.sh  –  Panorama-VLM multi-stage training pipeline
# ===============================================================

set -e          # 오류 발생 시 즉시 중단
set -o pipefail

CSV_TRAIN="data/train.csv"
CSV_VAL="data/val.csv"
IMAGE_ROOT="/data/quic360"
VISION_BCK="google/siglip-base-patch16-224"
LM_BCK="Qwen/Qwen3-0.6B"
NUM_WORKERS=8
BS=4

# ---------------- 1) Vision-only pre-train ---------------------
python -m train \
  --csv-train  "$CSV_TRAIN" \
  --csv-val    "$CSV_VAL"   \
  --image-root "$IMAGE_ROOT"\
  --vision-name  "$VISION_BCK" \
  --lm-name      "$LM_BCK" \
  --resampler    identity \
  --stage        vision \
  --epochs       1 \
  --batch-size   $BS \
  --num-workers  $NUM_WORKERS \
  --lr           5e-5

# ---------------- 2) Resampler + ITC/ITM -----------------------
python -m train \
  --csv-train  "$CSV_TRAIN" \
  --csv-val    "$CSV_VAL"   \
  --image-root "$IMAGE_ROOT"\
  --vision-name  "$VISION_BCK" \
  --lm-name      "$LM_BCK" \
  --resampler    conv \
  --stage        qformer \
  --epochs       1 \
  --batch-size   $BS \
  --num-workers  $NUM_WORKERS \
  --lr           1e-4

# ---------------- 3) Full fine-tune ----------------------------
python -m train \
  --csv-train  "$CSV_TRAIN" \
  --csv-val    "$CSV_VAL"   \
  --image-root "$IMAGE_ROOT"\
  --vision-name  "$VISION_BCK" \
  --lm-name      "$LM_BCK" \
  --resampler    conv \
  --stage        finetune \
  --epochs       1 \
  --batch-size   $BS \
  --num-workers  $NUM_WORKERS \
  --lr           2e-5