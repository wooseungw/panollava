#!/bin/bash

# =============================================================================
# Custom Training Script with Flexible Configuration
# =============================================================================

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --stage STAGE          Training stage (vision|resampler|finetune|all)"
    echo "  -e, --epochs EPOCHS        Number of epochs (default: auto per stage)"
    echo "  -b, --batch-size SIZE      Batch size (default: auto per stage)"
    echo "  -l, --lr RATE              Learning rate (default: auto per stage)"
    echo "  -r, --resume PATH          Resume from checkpoint"
    echo "  -d, --data-dir DIR         Data directory (default: data/quic360)"
    echo "  -w, --workers NUM          Number of workers (default: ${NUM_WORKERS})"
    echo "  -p, --project NAME         WandB project name (default: ${WANDB_PROJECT})"
    echo "  -n, --name NAME            WandB run name (default: auto-generated)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --stage vision --epochs 5 --batch-size 16"
    echo "  $0 --stage all --data-dir /path/to/data"
    echo "  $0 --stage finetune --resume runs/vlm_resampler/checkpoints/best.ckpt"
    echo ""
}

# Default values
STAGE="all"
EPOCHS=""
BATCH_SIZE=""
LEARNING_RATE=""
RESUME_FROM=""
DATA_DIR="data/quic360"
NUM_WORKERS=4
WANDB_PROJECT="panollava-training"
WANDB_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--stage)
            STAGE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -p|--project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -n|--name)
            WANDB_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate stage
if [[ ! "$STAGE" =~ ^(vision|resampler|finetune|all)$ ]]; then
    echo "Error: Invalid stage '$STAGE'. Must be one of: vision, resampler, finetune, all"
    exit 1
fi

# Set data paths
CSV_TRAIN="${DATA_DIR}/train.csv"
CSV_VAL="${DATA_DIR}/valid.csv"

# Validate data files
if [ ! -f "$CSV_TRAIN" ]; then
    echo "Error: Training data file not found: $CSV_TRAIN"
    exit 1
fi

if [ ! -f "$CSV_VAL" ]; then
    echo "Error: Validation data file not found: $CSV_VAL"
    exit 1
fi

# Model Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"

# Generate run name if not provided
if [ -z "$WANDB_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    WANDB_NAME="${STAGE}_${TIMESTAMP}"
fi

echo "========================================"
echo "PanoLLaVA Custom Training"
echo "========================================"
echo "Stage: $STAGE"
echo "Data directory: $DATA_DIR"
echo "Workers: $NUM_WORKERS"
echo "WandB project: $WANDB_PROJECT"
echo "WandB name: $WANDB_NAME"
if [ -n "$RESUME_FROM" ]; then
    echo "Resume from: $RESUME_FROM"
fi
echo "========================================"

# Create directories
mkdir -p logs
mkdir -p runs

# Build command
CMD="python train.py"

# Add stage configuration
if [ "$STAGE" = "all" ]; then
    CMD="$CMD --stages vision resampler finetune"
else
    CMD="$CMD --stage $STAGE"
fi

# Add optional parameters
if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD --lr $LEARNING_RATE"
fi

if [ -n "$RESUME_FROM" ]; then
    if [ ! -f "$RESUME_FROM" ]; then
        echo "Error: Checkpoint file not found: $RESUME_FROM"
        exit 1
    fi
    CMD="$CMD --resume-from $RESUME_FROM"
fi

# Add fixed parameters
CMD="$CMD --vision-name $VISION_MODEL"
CMD="$CMD --lm-name $LM_MODEL"
CMD="$CMD --resampler $RESAMPLER"
CMD="$CMD --csv-train $CSV_TRAIN"
CMD="$CMD --csv-val $CSV_VAL"
CMD="$CMD --crop-strategy e2p"
CMD="$CMD --num-workers $NUM_WORKERS"
CMD="$CMD --max-txt-len $MAX_TXT_LEN"
CMD="$CMD --image-size $IMAGE_SIZE"
CMD="$CMD --system-msg \"$FINETUNE_SYSTEM_MSG\""
CMD="$CMD --wandb-project $WANDB_PROJECT"
CMD="$CMD --wandb-name $WANDB_NAME"

# Execute command
echo "Executing: $CMD"
echo ""

LOG_FILE="logs/${WANDB_NAME}.log"
eval "$CMD" 2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "Training completed!"
echo "Log file: $LOG_FILE"
echo "========================================"