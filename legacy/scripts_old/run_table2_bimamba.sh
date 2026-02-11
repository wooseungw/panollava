#!/usr/bin/env bash
set -euo pipefail

# Table 2 Bi-Mamba ablation runner
# Requirements: pip install mamba-ssm causal-conv1d

CONFIG=${CONFIG:-configs/anyres_e2p_bimamba.yaml}
CSV_TEST=${CSV_TEST:-data/quic360/test.csv}

echo "Using config: ${CONFIG}"
echo "Test CSV: ${CSV_TEST}"

python scripts/train.py --config "${CONFIG}" --stage vision,resampler,finetune
python scripts/eval.py --config "${CONFIG}" --csv-input "${CSV_TEST}"

echo "Done. Check runs/ and eval outputs for metrics (BLEU/METEOR/ROUGE/SPICE/CIDEr)."
