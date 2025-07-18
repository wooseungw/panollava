# 공통 환경 변수 및 인자
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
VISION_MODEL="google/siglip-base-patch16-224"
LM_MODEL="Qwen/Qwen3-0.6B"
RESAMPLER="mlp"
CROP_STRATEGY="e2p"
CSV_TRAIN="data/quic360/train.csv"
CSV_VAL="data/quic360/valid.csv"
NUM_WORKERS=64
WANDB_PROJECT="panollava-training"
