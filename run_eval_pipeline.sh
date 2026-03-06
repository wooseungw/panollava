#!/bin/bash

# ==============================================================================
# PanoLLaVA Evaluation Pipeline Script
# ==============================================================================
# 이 스크립트는 CORA 폴더 내의 4가지 Ablation 모델을 병렬로 평가합니다.
# - 환경 변수 설정 (CUDA, PYTHONPATH)
# - CSV 경로 자동 수정
# - 배치 사이즈 최적화 (1 -> 64)
# - GPU 부하 분산 (0번, 1번 GPU)
# ==============================================================================

# 1. 기본 설정
PYTHON_EXEC="/data/3_lib/miniconda3/envs/pano/bin/python"
BASE_DIR="$(pwd)"
CORA_DIR="$BASE_DIR/CORA"
OUTPUT_DIR="$CORA_DIR/outputs"
DATA_DIR="$CORA_DIR/data/quic360"
CSV_ORIGINAL="$DATA_DIR/test.csv"
CSV_FIXED="$DATA_DIR/test_fixed.csv"

# 2. 환경 설정 (Mamba & PYTHONPATH)
echo "🔧 Setting up environment..."
export PYTHONPATH="$PYTHONPATH:$CORA_DIR"

# Mamba CUDA 라이브러리 경로 설정
CUDA_LIB_PATH=$(find /data/3_lib/miniconda3/envs/pano/lib/python*/site-packages/nvidia/cuda_runtime/lib -name "libcudart.so.12" -exec dirname {} \; 2>/dev/null | head -1)
if [ -n "$CUDA_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
    echo "✅ CUDA Lib Path added: $CUDA_LIB_PATH"
else
    echo "⚠️  Warning: CUDA 12 library not found. Mamba-SSM might fail."
fi

# 3. 데이터셋 경로 수정 (test.csv -> test_fixed.csv)
# 원본 CSV의 상대 경로(data/...)를 절대 경로에 가깝게(CORA/data/...) 수정
if [ ! -f "$CSV_FIXED" ]; then
    echo "📄 Creating fixed CSV file ($CSV_FIXED)..."
    if [ -f "$CSV_ORIGINAL" ]; then
        sed 's|^data/|CORA/data/|' "$CSV_ORIGINAL" > "$CSV_FIXED"
        echo "✅ CSV fixed."
    else
        echo "❌ Error: Original CSV not found at $CSV_ORIGINAL"
        exit 1
    fi
else
    echo "✅ Fixed CSV already exists."
fi

# 4. 평가 실행 함수
run_eval() {
    local MODEL_NAME=$1
    local GPU_ID=$2
    local BATCH_SIZE=${3:-64}
    
    local CHECKPOINT_DIR="$OUTPUT_DIR/$MODEL_NAME/finetune"
    local CONFIG_PATH="$CHECKPOINT_DIR/config.yaml"
    local TEMP_CONFIG="temp_config_${MODEL_NAME}.yaml"
    local LOG_FILE="eval_${MODEL_NAME}.log"

    # Config 파일 확인
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "❌ Config not found for $MODEL_NAME, skipping..."
        return
    fi

    echo "🚀 [GPU $GPU_ID] Starting evaluation for $MODEL_NAME..."
    
    # 배치 사이즈를 1에서 64로 수정한 임시 설정 파일 생성 (속도 향상)
    sed "s/eval_batch_size: 1/eval_batch_size: $BATCH_SIZE/" "$CONFIG_PATH" > "$TEMP_CONFIG"

    # 평가 스크립트 실행 (백그라운드)
    # nohup을 사용하지 않고 현재 세션 종속으로 실행 (스크립트 종료 시 함께 종료되지 않도록 주의하거나 wait 사용)
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_EXEC scripts/eval.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --csv-input "$CSV_FIXED" \
        --config "$TEMP_CONFIG" \
        > "$LOG_FILE" 2>&1 &
        
    local PID=$!
    echo "   Running (PID $PID) > $LOG_FILE"
}

# 5. 모델별 병렬 실행 (GPU 분산)
echo "=================================================="
echo "🏁 Starting Parallel Evaluations"
echo "=================================================="

# GPU 0 할당
run_eval "crop_ablation_anyres_e2p" 0 64
run_eval "crop_ablation_e2p" 0 64

# GPU 1 할당
run_eval "crop_ablation_cubemap" 1 64
run_eval "crop_ablation_resize" 1 64

# 6. 대기
echo "=================================================="
echo "⏳ All jobs launched. Waiting for completion..."
echo "   (Check log files: eval_*.log)"
echo "=================================================="

wait

echo "🎉 All evaluations finished successfully!"
rm temp_config_*.yaml 2>/dev/null
echo "🧹 Cleaned up temporary config files."
