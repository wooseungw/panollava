#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
# Simple training script - executes Python train.py with config
python train.py --config config_sig2qwen3.json --stages vision resampler finetune "$@"
python eval.py --config config_sig2qwen3.json --csv-input data/quic360/test.csv "$@"