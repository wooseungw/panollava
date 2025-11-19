# conda 설치 시
export JAVA_HOME=$CONDA_PREFIX

# apt 설치 시
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
#!/usr/bin/env bash
# Simple evaluation script - executes Python eval.py with config
# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=1

# python scripts/eval.py \
#     --csv-input results/eval_results/model/only_finetune.csv

python scripts/eval.py \
    --checkpoint runs/anyres_e2p_novision/finetune/anyres-e2p_bimamba/siglip2_bimamba_anyres-e2p_train_epoch00_loss2.4910.ckpt \
    --config runs/anyres_e2p_novision/finetune/anyres-e2p_bimamba/config.yaml
