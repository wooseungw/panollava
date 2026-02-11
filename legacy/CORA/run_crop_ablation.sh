#!/bin/bash
set -euo pipefail

# Usage:
#   ./CORA/run_crop_ablation.sh
#   CUDA_VISIBLE_DEVICES=0 ./CORA/run_crop_ablation.sh

PYTHON_BIN="/data/3_lib/miniconda3/envs/pano/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIGS=(
  "${REPO_ROOT}/CORA/configs/crop_ablation_resize.yaml"
  "${REPO_ROOT}/CORA/configs/crop_ablation_cubemap.yaml"
  "${REPO_ROOT}/CORA/configs/crop_ablation_e2p.yaml"
  "${REPO_ROOT}/CORA/configs/crop_ablation_anyres_e2p.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "----------------------------------------------------------------"
  echo "Running crop ablation config: ${config}"
  echo "----------------------------------------------------------------"
  "${PYTHON_BIN}" "${REPO_ROOT}/CORA/scripts/train.py" --config "${config}"
done

echo "----------------------------------------------------------------"
echo "All trainings completed. Starting evaluation..."
echo "----------------------------------------------------------------"

for config in "${CONFIGS[@]}"; do
  meta="$(${PYTHON_BIN} -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('${config}').read_text()); exp=cfg.get('experiment',{}).get('name',''); csv=cfg.get('data',{}).get('csv_test') or cfg.get('paths',{}).get('csv_test') or 'data/quic360/test.csv'; print(f'{exp}\t{csv}')")"
  exp_name="${meta%%$'\t'*}"
  csv_test_rel="${meta#*$'\t'}"

  checkpoint_path="${REPO_ROOT}/CORA/outputs/${exp_name}/finetune/last.ckpt"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[WARN] Checkpoint not found, skipping eval: ${checkpoint_path}"
    continue
  fi

  csv_test_path="${REPO_ROOT}/CORA/${csv_test_rel}"
  if [[ ! -f "${csv_test_path}" ]]; then
    echo "[WARN] Test CSV not found, skipping eval: ${csv_test_path}"
    continue
  fi

  echo "----------------------------------------------------------------"
  echo "Evaluating experiment: ${exp_name}"
  echo "Checkpoint: ${checkpoint_path}"
  echo "Test CSV: ${csv_test_path}"
  echo "----------------------------------------------------------------"

  "${PYTHON_BIN}" "${REPO_ROOT}/CORA/scripts/eval.py" \
    --checkpoint "${checkpoint_path}" \
    --csv "${csv_test_path}" \
    --save_predictions_csv
done

echo "----------------------------------------------------------------"
echo "Crop strategy ablation training + evaluation completed."
echo "----------------------------------------------------------------"
