#!/usr/bin/env bash
# Simple training script - relies on config.json for all settings
python train.py --config config.json --stages vision resampler finetune "$@"
python eval.py --config config.json --csv-input data/quic360/test.csv "$@"
