#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
# Simple training script - executes Python train.py with config
python train.py --config config.json --stages resampler finetune "$@"
python eval.py --config config.json --csv-input data/quic360/test.csv "$@"