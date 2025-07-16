export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
#!/bin/bash
# Resampler 스테이지 학습
python train.py --stage resampler --csv-train data/quic360/train.csv --csv-val data/quic360/valid.csv --vision-name google/siglip-base-patch16-224 --lm-name Qwen/Qwen3-0.6B --resampler mlp --epochs 1 --batch-size 16 --lr 2e-6 --max-txt-len 128 --wandb-project panorama-vlm
