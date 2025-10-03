#!/bin/bash
# Fix for mamba_ssm CUDA library path issue
# This script sets the correct LD_LIBRARY_PATH for CUDA 12 libraries

# Find CUDA 12 runtime library in conda environment
CUDA_LIB_PATH=$(find /data/3_lib/miniconda3/envs/pano/lib/python*/site-packages/nvidia/cuda_runtime/lib -name "libcudart.so.12" -exec dirname {} \; 2>/dev/null | head -1)

if [ -z "$CUDA_LIB_PATH" ]; then
    echo "❌ CUDA 12 runtime library not found in conda environment"
    echo "Please install: pip install nvidia-cuda-runtime-cu12"
    exit 1
fi

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"

echo "✅ CUDA library path set: $CUDA_LIB_PATH"
echo "You can now run Python scripts that use mamba_ssm"
echo ""
echo "Example:"
echo "  source fix_mamba_cuda.sh"
echo "  python your_script.py"
