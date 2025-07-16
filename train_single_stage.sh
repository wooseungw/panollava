#!/bin/bash
# 단일 스테이지 학습 (예: vision)
python train.py --stage vision --csv-train data/quic360/train.csv --csv-val data/quic360/valid.csv --vision-name google/siglip-base-patch16-224 --lm-name Qwen/Qwen3-0.6B --resampler mlp --epochs 1 --batch-size 4 --lr 5e-5 --max-txt-len 128 --wandb-project panorama-vlm
