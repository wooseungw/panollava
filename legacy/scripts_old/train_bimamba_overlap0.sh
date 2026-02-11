#!/bin/bash
# BiMamba Overlap=0 학습 스크립트 (Stage 2-3만)
# 
# 사용법:
#   bash scripts/train_bimamba_overlap0.sh

set -e  # 에러 발생 시 중단

echo "======================================================"
echo "BiMamba Resampler - Overlap=0 Training (Stage 2-3)"
echo "======================================================"

# 1단계 체크포인트 경로 (기존 학습된 것 사용)
# 옵션 1: BiMamba로 학습된 vision (권장)
VISION_CHECKPOINT="runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/vision/anyres-e2p_bimamba/best.ckpt"

# 옵션 2: MLP로 학습된 vision (대안)
# VISION_CHECKPOINT="runs/siglip2-so400m_Qwen3_mlp_anyres-e2p_PE/vision/anyres-e2p_mlp/best.ckpt"

# 체크포인트 존재 확인
if [ ! -f "$VISION_CHECKPOINT" ]; then
    echo "❌ 에러: Vision 체크포인트를 찾을 수 없습니다: $VISION_CHECKPOINT"
    echo ""
    echo "사용 가능한 vision 체크포인트를 확인하세요:"
    echo "  ls runs/*/vision/*/best.ckpt"
    echo ""
    echo "올바른 경로로 수정 후 다시 실행하세요."
    exit 1
fi

echo "✅ Vision 체크포인트 발견: $VISION_CHECKPOINT"
echo ""

# 학습 실행
echo "🚀 Stage 2 (Resampler) & Stage 3 (Finetune) 시작..."
echo "   - Resampler: BiMamba"
echo "   - Overlap: 0.0"
echo "   - Vision checkpoint: $VISION_CHECKPOINT"
echo ""

python scripts/train.py \
    --config configs/bimamba_overlap0.yaml \
    --resume "$VISION_CHECKPOINT"

echo ""
echo "======================================================"
echo "✅ 학습 완료!"
echo "======================================================"
echo ""
echo "결과 확인:"
echo "  - 체크포인트: runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/"
echo "  - 로그: runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/*/logs/"
echo ""
