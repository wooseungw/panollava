#!/bin/bash
# VLM 모델 ablation study 실행 스크립트
# 업데이트: sacrebleu + basic_cleanup 자동 적용
# 새로운 모델: gemma-3-4b, qwen2.5-vl-3b

export CUDA_VISIBLE_DEVICES=0

# FlashAttention 관련 호환성 문제 해결
# LLaVA-OneVision 모델이 flash_attn_varlen_func를 요구하지만
# 최신 transformers에는 없으므로 eager attention 사용
export DISABLE_FLASH_ATTN=1

# Conda 환경 활성화
source /data/3_lib/miniconda3/etc/profile.d/conda.sh
conda activate pano

# 설정
DATA_CSV="data/quic360/test.csv"
OUTPUT_DIR="results"
BATCH_SIZE=2
MAX_NEW_TOKENS=64
DEVICE="cuda"

# ============================================================
# 평가할 모델 리스트 (최신 업데이트 - 2025-10-13)
# ============================================================
# 최신 모델 (권장):
#   - gemma-3-4b: Google의 최신 VLM (4B, chat template)
#   - qwen2.5-vl-3b: Qwen 최신 버전 (3B, chat template + vision utils)
#   - llava-onevision-0.5b: LLaVA-OneVision (0.5B, chat template + vision utils)
#   - llava-onevision-4b: LLaVA-OneVision-1.5 (4B, chat template + vision utils)
#   - llava-onevision-7b: LLaVA-OneVision (7B, chat template + vision utils)
#   - florence-2-large: Microsoft Florence-2 (task token 기반)
#   - internvl3.5-1b/2b/4b/8b: InternVL3.5 시리즈 (1B~8B, dynamic preprocessing)
#
# 제거된 모델:
#   - qwen-vl-chat: qwen2.5-vl-3b로 업그레이드
#   - qwen2-vl-2b: qwen2.5-vl-3b로 업그레이드
#   - cogvlm2-llama3-chat-19b: 19B로 메모리 부담이 큼
#   - internvl2-2b: internvl3.5 시리즈로 업그레이드
#   - llama3-llava-next-8b: llava-onevision-7b로 업그레이드
# ============================================================

# 전체 모델 평가 (약 3-4시간 소요)
MODELS=(
    # "blip2-opt-2.7b"           # 2.7B - 가장 경량
    # "blip2-flan-t5-xl"         # 3B - BLIP2 Flan-T5 버전
    # "qwen2.5-vl-3b"            # 3B - Qwen 최신 (chat template)
    "gemma-3-4b"             # 4B - Google Gemma (VLM) ✅ 완료
    # "llava-1.5-7b"             # 7B - LLaVA 기본 버전
    # "llava-1.6-mistral-7b"     # 7B - LLaVA 1.6 (Mistral 기반)
    # "instructblip-vicuna-7b"   # 7B - InstructBLIP
    # "llava-onevision-0.5b"     # 0.5B - LLaVA-OneVision (가장 경량)
    "llava-onevision-4b"       # 4.7B - LLaVA-OneVision-1.5 (chat template)
    # "llava-onevision-7b"       # 7B - LLaVA-OneVision (chat template)
    # "internvl3.5-1b"           # 1.1B - InternVL3.5 (가장 경량)
    "internvl3.5-2b"           # 2.3B - InternVL3.5
    # "internvl3.5-4b"           # 4.7B - InternVL3.5
    # "internvl3.5-8b"           # 8.5B - InternVL3.5 (가장 강력)
    # "florence-2-large"         # 0.77B - Florence-2 (task token)
)

# 빠른 테스트용 (경량 모델만, 약 30분 소요)
# MODELS=(
#     "llava-onevision-0.5b"
#     "internvl3.5-1b"
#     "qwen2.5-vl-3b"
#     "gemma-3-4b"
#     "llava-onevision-4b"
#     "florence-2-large"
# )

# InternVL3.5 전체 시리즈 비교 (1B/2B/4B/8B)
# MODELS=(
#     "internvl3.5-1b"
#     "internvl3.5-2b"
#     "internvl3.5-4b"
#     "internvl3.5-8b"
# )

# 최신 모델만 테스트 (2025년 추가 모델)
# MODELS=(
#     "llava-onevision-0.5b"
#     "llava-onevision-4b"
#     "llava-onevision-7b"
#     "florence-2-large"
#     "internvl3.5-2b"
#     "internvl3.5-8b"
#     "qwen2.5-vl-3b"
#     "gemma-3-4b"
# )

# ============================================================
# 평가 실행
# ============================================================

echo "============================================================"
echo "VLM Ablation Study 시작"
echo "============================================================"
echo "총 모델 수: ${#MODELS[@]}"
echo "데이터: $DATA_CSV"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "배치 크기: $BATCH_SIZE"
echo "============================================================"
echo ""

# 평가할 모델 리스트 출력
echo "평가할 모델:"
for i in "${!MODELS[@]}"; do
    echo "  $((i+1)). ${MODELS[$i]}"
done
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)
COMPLETED=0
FAILED=0

# 각 모델에 대해 평가 실행
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i+1))
    TOTAL_MODELS=${#MODELS[@]}

    echo "============================================================"
    echo "[$MODEL_NUM/$TOTAL_MODELS] 평가 시작: $MODEL"
    echo "============================================================"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 모델 평가 실행
    if python scripts/evaluate_vlm_models.py \
        --data_csv "$DATA_CSV" \
        --models "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --device "$DEVICE"; then

        ((COMPLETED++))
        echo ""
        echo "✓ [$MODEL_NUM/$TOTAL_MODELS] $MODEL 평가 완료"
    else
        ((FAILED++))
        echo ""
        echo "✗ [$MODEL_NUM/$TOTAL_MODELS] $MODEL 평가 실패"
    fi

    echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"

    # GPU 메모리 정리 (마지막 모델이 아닐 때만)
    if [ $MODEL_NUM -lt $TOTAL_MODELS ]; then
        echo "GPU 메모리 정리 중..."
        sleep 5
    fi

    echo ""
done

# 종료 시간 및 요약
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "============================================================"
echo "VLM Ablation Study 완료"
echo "============================================================"
echo "총 소요 시간: ${ELAPSED_MIN}분 ${ELAPSED_SEC}초"
echo "완료된 모델: $COMPLETED/${#MODELS[@]}"
echo "실패한 모델: $FAILED/${#MODELS[@]}"
echo "결과 위치: $OUTPUT_DIR/"
echo "============================================================"
echo ""

# 결과 파일 확인
if [ -d "$OUTPUT_DIR" ]; then
    echo "생성된 결과 파일:"
    find "$OUTPUT_DIR" -name "*_metrics.json" -o -name "*_predictions.csv" | sort
fi
