export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
#!/bin/bash

# Panorama-VLM 3-stage 학습 스크립트 예시
# 각 스테이지별 하이퍼파라미터는 필요에 따라 수정 가능


# 공통 변수
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"
VISION_NAME="google/siglip-base-patch16-224"
LM_NAME="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"
PROJECT="panorama-vlm"
MAX_TXT_LEN=512
NUM_WORKERS=4

# 스테이지별 하이퍼파라미터
VISION_EPOCHS=1
VISION_LR=1e-5
VISION_BATCH=32
VISION_VICREG=1.0

RESAMP_EPOCHS=1
RESAMP_LR=2e-6
RESAMP_BATCH=16
RESAMP_VICREG=0.0

FINETUNE_EPOCHS=1
FINETUNE_LR=2e-6
FINETUNE_BATCH=16
FINETUNE_VICREG=0.0

# 1. VICReg(vision) stage
python train.py \
  --stage vision \
  --epochs $VISION_EPOCHS \
  --lr $VISION_LR \
  --batch-size $VISION_BATCH \
  --vicreg-loss-weight $VISION_VICREG \
  --csv-train "$CSV_TRAIN" \
  --csv-val "$CSV_VAL" \
  --vision-name "$VISION_NAME" \
  --lm-name "$LM_NAME" \
  --resampler "$RESAMPLER" \
  --crop-strategy e2p \
  --wandb-project "$PROJECT" \
  --max-txt-len $MAX_TXT_LEN \
  --num-workers $NUM_WORKERS

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
  --epochs $RESAMP_EPOCHS \
  --lr $RESAMP_LR \
  --batch-size $RESAMP_BATCH \
  --vicreg-loss-weight $RESAMP_VICREG \
  --csv-train "$CSV_TRAIN" \
  --csv-val "$CSV_VAL" \
  --vision-name "$VISION_NAME" \
  --lm-name "$LM_NAME" \
  --resampler "$RESAMPLER" \
  --crop-strategy e2p \
  --wandb-project "$PROJECT" \
  --max-txt-len $MAX_TXT_LEN \
  --num-workers $NUM_WORKERS \
  ${RESAMP_CKPT:+--resume-from "$RESAMP_CKPT"}

# 3. Finetune stage
FINETUNE_CKPT=$(ls -t ./runs/vlm_resampler/checkpoints/*.ckpt 2>/dev/null | head -n 1)
if [ -z "$FINETUNE_CKPT" ]; then
  echo "[WARN] No checkpoint found for finetune stage."
  FINETUNE_CKPT=""
else
  echo "[INFO] Using checkpoint for finetune: $FINETUNE_CKPT"
fi
python train.py \
  --stage finetune \
  --epochs $FINETUNE_EPOCHS \
  --lr $FINETUNE_LR \
  --batch-size $FINETUNE_BATCH \
  --vicreg-loss-weight $FINETUNE_VICREG \
  --csv-train "$CSV_TRAIN" \
  --csv-val "$CSV_VAL" \
  --vision-name "$VISION_NAME" \
  --lm-name "$LM_NAME" \
  --resampler "$RESAMPLER" \
  --crop-strategy e2p \
  --wandb-project "$PROJECT" \
  --max-txt-len $MAX_TXT_LEN \
  --num-workers $NUM_WORKERS \
  ${FINETUNE_CKPT:+--resume-from "$FINETUNE_CKPT"}

# 실행 전, 경로/하이퍼파라미터를 프로젝트에 맞게 수정하세요.
