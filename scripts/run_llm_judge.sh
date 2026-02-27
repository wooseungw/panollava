#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LLM-as-a-Judge evaluation wrapper
# Loads .env, validates OPENAI_API_KEY, runs llm_judge_eval.py
# ============================================================
#
# Usage:
#   ./scripts/run_llm_judge.sh <predictions.csv>
#   ./scripts/run_llm_judge.sh predictions.csv --model gpt-4o
#   ./scripts/run_llm_judge.sh predictions.csv --max-samples 50 --save-stats
#
# All extra arguments are forwarded to llm_judge_eval.py.
# Default model: gpt-4.1-mini (cost-effective).
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------- Load .env from project root ----------
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---------- Validate API key ----------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo ""
    echo "Set it in one of:"
    echo "  1. $PROJECT_ROOT/.env   (OPENAI_API_KEY=sk-...)"
    echo "  2. export OPENAI_API_KEY=sk-..."
    echo "  3. Pass --api-key sk-... as argument"
    exit 1
fi

# ---------- Validate input ----------
if [[ $# -lt 1 ]] || [[ "$1" == -* ]]; then
    echo "Usage: $0 <predictions.csv> [options]"
    echo ""
    echo "Options (forwarded to llm_judge_eval.py):"
    echo "  --model MODEL        OpenAI model (default: gpt-4.1-mini)"
    echo "  --output PATH        Output CSV path"
    echo "  --max-samples N      Evaluate first N samples"
    echo "  --no-image           Text-only evaluation"
    echo "  --batch-by-image     Group by image for fewer API calls"
    echo "  --save-stats         Save .stats.json"
    echo "  --base-path DIR      Base dir for relative image paths"
    echo "  --api-key KEY        Override OPENAI_API_KEY"
    echo ""
    echo "Examples:"
    echo "  $0 outputs/predictions.csv"
    echo "  $0 outputs/predictions.csv --model gpt-4o --batch-by-image"
    echo "  $0 outputs/predictions.csv --max-samples 50 --save-stats"
    exit 1
fi

INPUT_FILE="$1"
shift

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# ---------- Default model (only add if not already specified) ----------
MODEL="${MODEL:-gpt-4.1-mini}"

HAS_MODEL=false
for arg in "$@"; do
    if [[ "$arg" == "--model" ]] || [[ "$arg" == "-m" ]]; then
        HAS_MODEL=true
        break
    fi
done

EXTRA_ARGS=()
if [[ "$HAS_MODEL" == false ]]; then
    EXTRA_ARGS+=(--model "$MODEL")
fi

# ---------- Run ----------
echo "============================================================"
echo "  LLM Judge Evaluation"
echo "  Input : $INPUT_FILE"
echo "  Model : $MODEL"
echo "============================================================"

exec python "$SCRIPT_DIR/llm_judge_eval.py" \
    --input "$INPUT_FILE" \
    "${EXTRA_ARGS[@]}" \
    "$@"
