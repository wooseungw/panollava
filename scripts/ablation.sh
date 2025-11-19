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



# Run training (stages are read from YAML config)
python "${PROJECT_ROOT}/scripts/train.py" --config configs/resize.yaml
python "${PROJECT_ROOT}/scripts/train.py" --config configs/cubemap.yaml
python "${PROJECT_ROOT}/scripts/train.py" --config configs/silding.yaml


