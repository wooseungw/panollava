#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM-as-a-Judge Evaluation Script for Panorama VLM (GPT-5.2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 사용법:
#   ./scripts/run_llm_judge.sh <predictions.csv 경로>
#
# 예시:
#   ./scripts/run_llm_judge.sh results/eval_results/.../predictions.csv
#
# 특징:
#   - 같은 이미지의 샘플들을 배치로 묶어 한 번에 평가 (API 호출 ~16배 감소)
#   - 5961개 샘플 → 374개 API 호출 (이미지별 배치)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# =====================
# 설정 (필요시 수정)
# =====================
MODEL="gpt-5-mini"          # gpt-5.2, gpt-5-mini (권장), gpt-5.2-pro
REASONING="low"             # none, low, medium, high (low 권장 - 비용 절약)
VERBOSITY="low"             # low, medium, high
BATCH_BY_IMAGE=true         # 이미지별 배치 평가 (API 호출 감소)
SAVE_STATS=true             # 통계 JSON 저장 여부

# =====================
# .env 파일 로드
# =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "📁 Loading .env from $PROJECT_ROOT/.env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# =====================
# OpenAI API 키 확인
# =====================
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
    echo ""
    echo "설정 방법:"
    echo "  1. .env 파일에 추가: OPENAI_API_KEY=your-api-key"
    echo "  2. 또는 export OPENAI_API_KEY='your-api-key'"
    echo ""
    exit 1
fi

# =====================
# 입력 확인
# =====================
if [ $# -lt 1 ]; then
    echo "사용법: $0 <predictions.csv 경로>"
    echo ""
    echo "예시:"
    echo "  $0 results/eval_results/.../predictions.csv"
    echo ""
    exit 1
fi

CSV_INPUT="$1"

if [ ! -f "$CSV_INPUT" ]; then
    echo "❌ Error: 입력 파일을 찾을 수 없습니다: $CSV_INPUT"
    exit 1
fi

# =====================
# 평가 실행
# =====================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 LLM-as-a-Judge Evaluation (GPT-5.2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📄 입력: $CSV_INPUT"
echo "  🤖 모델: $MODEL"
echo "  🧠 Reasoning: $REASONING"
echo "  📝 Verbosity: $VERBOSITY"
echo "  📦 이미지별 배치: $BATCH_BY_IMAGE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# =====================
# 평가 실행
# =====================
CMD="python $SCRIPT_DIR/llm_judge_eval.py \
    --csv-input \"$CSV_INPUT\" \
    --model $MODEL \
    --reasoning $REASONING \
    --verbosity $VERBOSITY \
    --base-path \"$PROJECT_ROOT\""

if [ "$BATCH_BY_IMAGE" = true ]; then
    CMD="$CMD --batch-by-image"
fi

if [ "$SAVE_STATS" = true ]; then
    CMD="$CMD --save-stats"
fi

echo "실행: $CMD"
echo ""

eval $CMD

echo ""
echo "✅ LLM Judge 평가 완료!"
