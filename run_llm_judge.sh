#!/usr/bin/env bash
# LLM-as-a-Judge evaluation for all main-table models
# Models: InternVL3.5-2B, Qwen2.5-VL-3B, Gemma3-4B × {Resize, Native, DenseCL, VICReg-pw}
# Output: runs/baseline/{model}/llm_judge/predictions_judge_scores.{csv,stats.json}
set -e

WORKDIR=/data/1_personal/4_SWWOO/panollava
PYTHON=/home/wsw/miniconda3/envs/pano/bin/python
cd "$WORKDIR"

# Load OPENAI_API_KEY from .env
if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] OPENAI_API_KEY not set"
    exit 1
fi

# Judge config
JUDGE_MODEL="gpt-5.2"

# ── Main-table models (paper Tab. 1) ─────────────────────────────────────
# dir_name → display label (for logging)
declare -A MODELS
MODELS["native_internvl35-2b"]="InternVL3.5-2B / Native"
MODELS["native_qwen25-vl-3b"]="Qwen2.5-VL-3B / Native"
MODELS["native_gemma3-4b"]="Gemma3-4B / Native"
MODELS["panoadapt_internvl35-2b"]="InternVL3.5-2B / +DenseCL"
MODELS["panoadapt_qwen25-vl-3b"]="Qwen2.5-VL-3B / +DenseCL"
MODELS["panoadapt_gemma3-4b"]="Gemma3-4B / +DenseCL"
MODELS["panoadapt_vicreg_pairwise_internvl35-2b"]="InternVL3.5-2B / +VICReg-pw"
MODELS["panoadapt_vicreg_pairwise_qwen25-vl-3b"]="Qwen2.5-VL-3B / +VICReg-pw"
MODELS["internvl3_5-2b_img256_tok128"]="InternVL3.5-2B / Resize"
MODELS["qwen25-vl-3b_img256_tok128"]="Qwen2.5-VL-3B / Resize"
MODELS["gemma3-4b_img256_tok128"]="Gemma3-4B / Resize"

# Fixed evaluation order (matches paper table order)
ORDER=(
    "internvl3_5-2b_img256_tok128"
    "native_internvl35-2b"
    "panoadapt_internvl35-2b"
    "panoadapt_vicreg_pairwise_internvl35-2b"
    "qwen25-vl-3b_img256_tok128"
    "native_qwen25-vl-3b"
    "panoadapt_qwen25-vl-3b"
    "panoadapt_vicreg_pairwise_qwen25-vl-3b"
    "gemma3-4b_img256_tok128"
    "native_gemma3-4b"
    "panoadapt_gemma3-4b"
)

TOTAL=${#ORDER[@]}
IDX=0

for dir_name in "${ORDER[@]}"; do
    IDX=$((IDX + 1))
    label="${MODELS[$dir_name]}"
    input="runs/baseline/${dir_name}/eval/predictions.csv"
    out_dir="runs/baseline/${dir_name}/llm_judge"
    output="${out_dir}/predictions_judge_scores.csv"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$IDX/$TOTAL] ${label}"
    echo "  dir: ${dir_name}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════"

    if [ ! -f "$input" ]; then
        echo "  [SKIP] predictions.csv not found: $input"
        continue
    fi

    # Skip if stats already exist (resume-safe)
    stats_file="${out_dir}/predictions_judge_scores.stats.json"
    if [ -f "$stats_file" ]; then
        echo "  [SKIP] Stats already exist: $stats_file"
        continue
    fi

    mkdir -p "$out_dir"

    $PYTHON scripts/llm_judge_eval.py \
        --input          "$input"  \
        --output         "$output" \
        --model          "$JUDGE_MODEL" \
        --use-batch-api  \
        --image-max-px   1024 \
        --dedup          \
        --sample-n       300 \
        --sample-seed    42 \
        --save-stats

    echo "  [DONE] Saved: $stats_file"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  All LLM-Judge evaluations complete."
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"
