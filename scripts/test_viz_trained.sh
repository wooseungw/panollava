#!/bin/bash
# 학습된 모델 Vision Encoder 시각화 테스트 스크립트
# YAML 파일 기반으로 동작

set -e  # 에러 발생 시 중단

echo "=========================================="
echo "학습된 모델 Vision Encoder 시각화 테스트"
echo "=========================================="

# PYTHONPATH 설정
export PYTHONPATH=/data/1_personal/4_SWWOO/panollava/src:$PYTHONPATH

# 필요한 패키지 확인
echo "📦 필수 패키지 확인 중..."
python -c "import torch; print('  ✅ torch')" || { echo "  ❌ torch가 필요합니다"; exit 1; }
python -c "import numpy; print('  ✅ numpy')" || { echo "  ❌ numpy가 필요합니다"; exit 1; }
python -c "import matplotlib; print('  ✅ matplotlib')" || { echo "  ❌ matplotlib가 필요합니다"; exit 1; }
python -c "import PIL; print('  ✅ PIL')" || { echo "  ❌ PIL이 필요합니다"; exit 1; }
python -c "import sklearn; print('  ✅ sklearn')" || { echo "  ❌ sklearn이 필요합니다"; exit 1; }
python -c "import skimage; print('  ✅ skimage')" || { echo "  ⚠️  skimage 권장 (pip install scikit-image)"; }

# 기본 설정 (YAML 기반)
CONFIG="${1:-configs/default.yaml}"
CHECKPOINT="${2:-runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt}"
IMAGE="${3:-data/quic360/downtest/images/2094501355_045ede6d89_k.jpg}"
OUTPUT_DIR="${4:-results/trained_viz_test}"

echo ""
echo "🔧 설정:"
echo "  Config: $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Image: $IMAGE"
echo "  Output: $OUTPUT_DIR"
echo ""

# 파일 존재 확인
if [ ! -f "$CONFIG" ]; then
    echo "❌ 설정 파일을 찾을 수 없습니다: $CONFIG"
    echo ""
    echo "사용 가능한 설정 파일:"
    find configs -name "*.yaml" -type f
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 체크포인트를 찾을 수 없습니다: $CHECKPOINT"
    echo ""
    echo "사용 가능한 체크포인트:"
    find runs -name "*.ckpt" -type f | head -5
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "❌ 이미지를 찾을 수 없습니다: $IMAGE"
    echo ""
    echo "사용 가능한 이미지:"
    find data/quic360/downtest/images -name "*.jpg" -type f | head -3
    exit 1
fi

# 시각화 실행 (YAML 기반)
echo "🚀 시각화 시작..."
echo ""

python scripts/visualize_trained_model.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --image "$IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --device auto

echo ""
echo "=========================================="
echo "✅ 완료! 결과 확인: $OUTPUT_DIR"
echo "=========================================="
