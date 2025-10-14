#!/usr/bin/env bash
# Simple evaluation script - executes Python eval.py with config
python scripts/eval.py --config configs/default.yaml --csv-input data/quic360/test.csv "$@"