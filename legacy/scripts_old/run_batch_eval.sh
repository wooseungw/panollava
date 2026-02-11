#!/bin/bash

export CUDA_VISIBLE_DEVICES=0   

# Conda 환경 활성화 (필요 시 수정)
source /data/3_lib/miniconda3/etc/profile.d/conda.sh
conda activate pano


# CSV 파일 경로 (필요 시 수정)
CSV_INPUT="data/quic360/test.csv"

echo "========================================================================"
echo "🚀 Starting Batch Evaluation"
echo "========================================================================"

# 1. Finetune Checkpoint Evaluation
# config.yaml이 같은 디렉토리에 있으므로 --config 생략 가능 (자동 감지)
echo "------------------------------------------------------------------------"
echo "1️⃣  Evaluating Finetune Checkpoint..."
echo "------------------------------------------------------------------------"
python scripts/eval.py \
    --checkpoint runs/siglip2-so400m_Qwen306_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/siglip2_bimamba_anyres-e2p_train_epoch00_loss2.2804.ckpt \
    --csv-input "$CSV_INPUT"

echo ""
echo "------------------------------------------------------------------------"
echo "2️⃣  Evaluating Resampler Checkpoint..."
echo "------------------------------------------------------------------------"
# Resampler Only (LLM might be frozen/untrained depending on stage)
python scripts/eval.py \
    --checkpoint runs/siglip2-so400m_Qwen306_bimamba_anyres-e2p_PE/resampler/anyres-e2p_bimamba/siglip2_bimamba_anyres-e2p_train_epoch00_loss2.2723.ckpt \
    --csv-input "$CSV_INPUT"

echo ""
echo "------------------------------------------------------------------------"
echo "3️⃣  Evaluating Vision Checkpoint..."
echo "------------------------------------------------------------------------"
# Vision Pretrain (Typically frozen LLM, but evaluating VLM capabilties)
python scripts/eval.py \
    --checkpoint runs/siglip2-so400m_Qwen306_bimamba_anyres-e2p_PE/vision/anyres-e2p_bimamba/siglip2_bimamba_anyres-e2p_train_plus2_epoch02_loss6.0191.ckpt \
    --csv-input "$CSV_INPUT"

echo ""
echo "========================================================================"
echo "✅ Batch Evaluation Completed!"
echo "========================================================================"
