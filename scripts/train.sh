#!/usr/bin/env bash
# Simple training script - relies on config.yaml for all settings

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pano

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

# Default config file
CONFIG_FILE="${1:-configs/default.yaml}"

echo "Starting PanoLLaVA training pipeline..."
echo "Config: $CONFIG_FILE"
echo "Stages will be read from config.yaml (training.stages)"
echo ""

# Run training (stages are read from YAML config)
python "${PROJECT_ROOT}/scripts/train.py" --config "$CONFIG_FILE"

echo ""
echo "Training completed. Run evaluation separately if needed:"
echo "  python scripts/eval.py --config $CONFIG_FILE --csv-input data/quic360/test.csv"
