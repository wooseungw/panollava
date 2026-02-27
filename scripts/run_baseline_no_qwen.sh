#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Baseline Training + Evaluation Pipeline (QUIC-360)
# Non-Qwen models only (gemma3, blip2, internvl x2)
# Multi-model sequential ablation run — single GPU (GPU 1)
# ============================================================
# Usage:
#   bash scripts/run_baseline_no_qwen.sh                                           # all models
#   bash scripts/run_baseline_no_qwen.sh --max-tokens 256 --image-size 448 --max-length 1024
#   bash scripts/run_baseline_no_qwen.sh --models "gemma3-4b,blip2-flan-t5-xl"      # subset
# ============================================================

# Pin to GPU 1 only
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAX_TOKENS=128
IMAGE_SIZE=256
MAX_LENGTH=1024
SELECTED_MODELS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --image-size) IMAGE_SIZE="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --models)     SELECTED_MODELS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/run_baseline_no_qwen.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-tokens N   Max new tokens for generation (default: 64)"
            echo "  --image-size N   Image size in pixels per side (default: 256)"
            echo "  --max-length N   Max input sequence length / truncation (default: 1024)"
            echo "  --models LIST    Comma-separated model names to run (default: all)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Available models:"
            echo "  gemma3-4b         google/gemma-3-4b-it             (model_type: gemma3)"
            echo "  internvl3_5-1b    OpenGVLab/InternVL3_5-1B         (model_type: internvl)"
            echo "  internvl3_5-2b    OpenGVLab/InternVL3_5-2B         (model_type: internvl)"
            echo "  blip2-opt-2.7b    Salesforce/blip2-opt-2.7b        (model_type: blip2)"
            echo ""
            echo "Example:"
            echo "  bash scripts/run_baseline_no_qwen.sh --models 'gemma3-4b,blip2-flan-t5-xl'"
            exit 0
            ;;
        *) echo "Unknown arg: $1. Use --help for usage."; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUIC360_DIR="/data/1_personal/4_SWWOO/refer360/data/quic360_format"
TRAIN_CSV_RAW="$QUIC360_DIR/train.csv"
TEST_CSV_RAW="$QUIC360_DIR/test.csv"

# ============================================================
# Model definitions: NAME|HF_MODEL_ID|MODEL_TYPE|DTYPE
# ============================================================
ALL_MODELS=(
    "gemma3-4b|google/gemma-3-4b-it|gemma3|bfloat16"
    "internvl3_5-1b|OpenGVLab/InternVL3_5-1B|internvl|bfloat16"
    "internvl3_5-2b|OpenGVLab/InternVL3_5-2B|internvl|bfloat16"
    "blip2-opt-2.7b|Salesforce/blip2-opt-2.7b|blip2|bfloat16"
)

# Filter models if --models was specified
if [[ -n "$SELECTED_MODELS" ]]; then
    FILTERED=()
    IFS=',' read -ra NAMES <<< "$SELECTED_MODELS"
    for name in "${NAMES[@]}"; do
        name="$(echo "$name" | xargs)"  # trim whitespace
        found=false
        for entry in "${ALL_MODELS[@]}"; do
            entry_name="${entry%%|*}"
            if [[ "$entry_name" == "$name" ]]; then
                FILTERED+=("$entry")
                found=true
                break
            fi
        done
        if [[ "$found" == false ]]; then
            echo "ERROR: Unknown model '$name'. Use --help to see available models."
            exit 1
        fi
    done
    ALL_MODELS=("${FILTERED[@]}")
fi

echo "============================================================"
echo "  Baseline Pipeline: QUIC-360 (Non-Qwen Models)"
echo "  Image size : ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Max tokens : ${MAX_TOKENS}"
echo "  Max length : ${MAX_LENGTH}"
echo "  Models (${#ALL_MODELS[@]}):"
for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r m_name m_hf m_type m_dtype <<< "$entry"
    echo "    - ${m_name}  (${m_hf})"
done
echo "============================================================"

# ---------- Step 1: Prepare data (shared across models) ----------
echo "[1/3] Preparing QUIC-360 data..."

SHARED_DATA_DIR="$PROJECT_ROOT/runs/baseline/_shared_data"
mkdir -p "$SHARED_DATA_DIR"

python -c "
import pandas as pd
from pathlib import Path

for split, csv_raw in [('train', '${TRAIN_CSV_RAW}'), ('test', '${TEST_CSV_RAW}')]:
    df = pd.read_csv(csv_raw)
    df = df.rename(columns={'query': 'instruction', 'annotation': 'response'})
    df = df[df['url'].apply(lambda p: Path(p).is_file())]
    out = '${SHARED_DATA_DIR}/' + split + '.csv'
    df.to_csv(out, index=False)
    print(f'  {split}: {len(df)} samples -> {out}')
"

TRAIN_CSV="$SHARED_DATA_DIR/train.csv"
TEST_CSV="$SHARED_DATA_DIR/test.csv"

# ---------- Step 2: Loop over each model ----------
TOTAL=${#ALL_MODELS[@]}
CURRENT=0
FAILED_MODELS=()
SUCCEEDED_MODELS=()

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME HF_MODEL_ID MODEL_TYPE MODEL_DTYPE <<< "$entry"
    CURRENT=$((CURRENT + 1))
    OUTPUT_DIR="$PROJECT_ROOT/runs/baseline/${MODEL_NAME}_img${IMAGE_SIZE}_tok${MAX_TOKENS}"

    echo ""
    echo "============================================================"
    echo "  [${CURRENT}/${TOTAL}] ${MODEL_NAME}"
    echo "  HF ID      : ${HF_MODEL_ID}"
    echo "  Model type  : ${MODEL_TYPE}"
    echo "  Dtype       : ${MODEL_DTYPE}"
    echo "  Output      : ${OUTPUT_DIR}"
    echo "============================================================"

    mkdir -p "$OUTPUT_DIR"

    CONFIG_PATH="$OUTPUT_DIR/config.yaml"
    MIXED_PREC="${MODEL_DTYPE/float16/fp16}"
    MIXED_PREC="${MIXED_PREC/bfloat16/bf16}"
    cat > "$CONFIG_PATH" <<EOF
experiment_name: "baseline_${MODEL_NAME}_img${IMAGE_SIZE}_tok${MAX_TOKENS}"
output_dir: "${OUTPUT_DIR}"
max_new_tokens: ${MAX_TOKENS}

model:
  name: "${MODEL_NAME}"
  hf_model_id: "${HF_MODEL_ID}"
  model_type: "${MODEL_TYPE}"
  dtype: "${MODEL_DTYPE}"
  image_size: ${IMAGE_SIZE}

lora:
  r: 32
  alpha: 64
  dropout: 0.1

training:
  num_epochs: 1
  batch_size: -1
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_length: ${MAX_LENGTH}
  seed: 42
  gradient_checkpointing: true
  mixed_precision: "${MIXED_PREC}"
  logging_steps: 10
  save_strategy: "epoch"
  save_total_limit: 2
  dataloader_num_workers: 4

data:
  image_column: "url"
  instruction_column: "instruction"
  response_column: "response"

data_train_csv: "${TRAIN_CSV}"
data_test_csv: "${TEST_CSV}"

wandb_enabled: false
EOF

    echo "  Config: $CONFIG_PATH"

    # Train
    echo "------------------------------------------------------------"
    echo "  TRAINING: ${MODEL_NAME}"
    echo "------------------------------------------------------------"

    if python "$SCRIPT_DIR/baseline_finetune.py" --config "$CONFIG_PATH"; then
        # Evaluate (only if training succeeded)
        echo "------------------------------------------------------------"
        echo "  EVALUATION: ${MODEL_NAME}"
        echo "------------------------------------------------------------"

        if python "$SCRIPT_DIR/baseline_eval.py" \
            --config "$CONFIG_PATH" \
            --test-csv "$TEST_CSV" \
            --output-dir "$OUTPUT_DIR/eval"; then
            SUCCEEDED_MODELS+=("$MODEL_NAME")
        else
            echo "WARNING: Evaluation failed for ${MODEL_NAME}. Continuing..."
            FAILED_MODELS+=("${MODEL_NAME} (eval)")
        fi
    else
        echo "WARNING: Training failed for ${MODEL_NAME}. Skipping evaluation..."
        FAILED_MODELS+=("${MODEL_NAME} (train)")
    fi
done

# ---------- Summary ----------
echo ""
echo "============================================================"
echo "  ALL DONE — Summary"
echo "============================================================"
echo "  Succeeded (${#SUCCEEDED_MODELS[@]}/${TOTAL}):"
for m in "${SUCCEEDED_MODELS[@]}"; do
    echo "    ✓ $m  →  runs/baseline/${m}_img${IMAGE_SIZE}_tok${MAX_TOKENS}/eval/"
done
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed (${#FAILED_MODELS[@]}/${TOTAL}):"
    for m in "${FAILED_MODELS[@]}"; do
        echo "    ✗ $m"
    done
fi
echo "============================================================"
