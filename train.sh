#!/usr/bin/env bash
###############################################################################
# run_panovlm.sh  – Panorama-VLM 3-Stage 학습 파이프라인
# 사용 예시
#   $ chmod +x run_panovlm.sh
#   $ ./run_panovlm.sh train.csv val.csv /data/quic360 <GPU_ID>
###############################################################################

set -e                                  # 오류 발생 시 즉시 종료
######################### 0) 공통 인자 #######################################
CSV_TRAIN="$1"      # ex) train.csv
CSV_VAL="$2"        # ex) val.csv
GPU_ID="0"    # GPU 번호(기본 0)

EPOCH_VISION=1
EPOCH_RESAMP=1
EPOCH_FINETUNE=2
BATCH=4
MAX_TXT_LEN=2048
LR=5e-5
VISION_MODEL="openai/clip-vit-base-patch32"
LM_MODEL="Qwen/Qwen3-0.6B"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

mkdir -p logs checkpoints
###############################################################################
function run_stage () {
    local STAGE=$1
    local EPOCHS=$2
    local RESUME=$3     # ckpt 경로 (없으면 빈 문자열)
    local WARM=$4       # true|false

    NAME="${RUN_PREFIX}_${STAGE}"
    EXTRA=""

    if [[ -n "$RESUME" ]]; then
        EXTRA+=" --resume-from $RESUME"
        if [[ "$WARM" == "true" ]]; then
            EXTRA+=" --warm-start"
        fi
    fi

    python train.py \
        --csv-train "$CSV_TRAIN" \
        --csv-val "$CSV_VAL" \
        --vision-name "$VISION_MODEL" \
        --lm-name "$LM_MODEL" \
        --resampler mlp \
        --stage "$STAGE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --lr "$LR" \
        --max-txt-len "$MAX_TXT_LEN" \
        --wandb-name "$NAME" \
        $EXTRA \
        2>&1 | tee "logs/${NAME}.log"
}

######################### 1) Vision 단계 ######################################
run_stage vision   $EPOCH_VISION ""        false
CKPT_VISION=$(ls runs/vlm_vision/**/checkpoints/*.ckpt | head -n1)

######################### 2) Resampler 단계 ###################################
run_stage resampler $EPOCH_RESAMP "$CKPT_VISION" true
CKPT_RESAMP=$(ls runs/vlm_resampler/**/checkpoints/*.ckpt | head -n1)

######################### 3) Finetune 단계 ####################################
run_stage finetune $EPOCH_FINETUNE "$CKPT_RESAMP" true

echo "=== 모든 단계 완료 ==="
