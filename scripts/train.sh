#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
# Simple training script - executes Python train.py with config
python train.py --config config.json --stages vision resampler finetune "$@"