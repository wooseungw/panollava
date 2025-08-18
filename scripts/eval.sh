#!/usr/bin/env bash
# Simple evaluation script - executes Python eval.py with config
python eval.py --config config.json --csv-input data/quic360/test.csv "$@"