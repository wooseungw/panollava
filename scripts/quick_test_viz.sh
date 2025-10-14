#!/bin/bash
# 빠른 테스트 스크립트

export PYTHONPATH=/data/1_personal/4_SWWOO/panollava/src:$PYTHONPATH

CHECKPOINT="runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt"
IMAGE="data/quic360/downtest/images/2094501355_045ede6d89_k.jpg"

echo "🚀 학습된 모델 시각화 테스트"
echo "Checkpoint: $CHECKPOINT"
echo "Image: $IMAGE"

python scripts/visualize_trained_model.py \
    --checkpoint "$CHECKPOINT" \
    --image "$IMAGE" \
    --output_dir results/quick_test \
    --device cuda \
    --crop_strategy e2p

echo "✅ 완료!"
