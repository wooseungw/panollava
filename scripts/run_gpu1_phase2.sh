#!/bin/bash
# ==========================================================
# GPU 1 Phase 2 — VICReg-pairwise experiments + InternVL eval
# Runs AFTER run_gpu1_trackb.sh (B1/B3 DenseCL) completes
#
# Queue order:
#   1. VICReg-pw 25% InternVL eval   (trained, ~40min)
#   2. VICReg-pw 50% InternVL retrain + eval (~2.5h)
#   3. VICReg-pw 50% Qwen train + eval (~2.5h)
#   4. VICReg-pw 50% Gemma3 train + eval (~2.5h)
# ==========================================================
set -o pipefail

PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

SUCCEEDED=()
FAILED=()

run_task() {
    local label="$1"
    shift
    echo ""
    echo "=== $label ==="
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S KST')"
    if "$@"; then
        echo "  ✅ $label DONE ($(date '+%Y-%m-%d %H:%M:%S KST'))"
        SUCCEEDED+=("$label")
        return 0
    else
        echo "  ❌ $label FAILED (exit=$?, $(date '+%Y-%m-%d %H:%M:%S KST'))"
        FAILED+=("$label")
        return 1
    fi
}

echo "=========================================="
echo "  GPU 1 Phase 2 — VICReg-pairwise Experiments"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# ----------------------------------------------------------
# 1. VICReg-pw 25% InternVL — EVAL ONLY (already trained)
# ----------------------------------------------------------
echo ""
echo ">>> [1/4] VICReg-pw 25% InternVL eval (already trained)"
VICREG25_DIR="runs/baseline/panoadapt_vicreg_pairwise_internvl35-2b_25overlap"

if [ -d "${VICREG25_DIR}/lora_adapter" ]; then
    run_task "[VICReg-pw 25% InternVL] eval" \
        $PYTHON scripts/baseline_eval.py \
        --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b_25overlap.yaml \
        --output-dir "${VICREG25_DIR}/eval"
else
    echo "  ⚠️ No lora_adapter found at ${VICREG25_DIR} — SKIPPING"
    FAILED+=("[VICReg-pw 25% InternVL] no adapter")
fi

# ----------------------------------------------------------
# 2. VICReg-pw 50% InternVL — RETRAIN + EVAL
# ----------------------------------------------------------
echo ""
echo ">>> [2/4] VICReg-pw 50% InternVL retrain + eval"

# Smoke test first
$PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
cfg.training.num_epochs = 0.001
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST VICReg-pw 50% InternVL PASSED')
" 2>&1 | tee /tmp/smoke_vicreg_pw50_internvl.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ SMOKE TEST FAILED — check /tmp/smoke_vicreg_pw50_internvl.log"
    FAILED+=("[VICReg-pw 50% InternVL] smoke test")
else
    echo "  ✅ SMOKE TEST PASSED"

    # Clean previous incomplete run
    rm -rf runs/baseline/panoadapt_vicreg_pairwise_internvl35-2b/checkpoints/*
    rm -rf runs/baseline/panoadapt_vicreg_pairwise_internvl35-2b/final
    rm -rf runs/baseline/panoadapt_vicreg_pairwise_internvl35-2b/lora_adapter

    run_task "[VICReg-pw 50% InternVL] training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml

    if [[ ! "${FAILED[*]}" =~ "VICReg-pw 50% InternVL.*training" ]]; then
        run_task "[VICReg-pw 50% InternVL] eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml \
            --output-dir runs/baseline/panoadapt_vicreg_pairwise_internvl35-2b/eval
    fi
fi

# ----------------------------------------------------------
# 3. VICReg-pw 50% Qwen2.5-VL-3B — TRAIN + EVAL
# ----------------------------------------------------------
echo ""
echo ">>> [3/4] VICReg-pw 50% Qwen2.5-VL-3B train + eval"

$PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('configs/baseline/panoadapt_vicreg_pairwise_qwen25_3b.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
cfg.training.num_epochs = 0.001
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST VICReg-pw 50% Qwen PASSED')
" 2>&1 | tee /tmp/smoke_vicreg_pw50_qwen.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ SMOKE TEST FAILED — check /tmp/smoke_vicreg_pw50_qwen.log"
    FAILED+=("[VICReg-pw 50% Qwen] smoke test")
else
    echo "  ✅ SMOKE TEST PASSED"
    run_task "[VICReg-pw 50% Qwen] training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_vicreg_pairwise_qwen25_3b.yaml

    if [[ ! "${FAILED[*]}" =~ "VICReg-pw 50% Qwen.*training" ]]; then
        run_task "[VICReg-pw 50% Qwen] eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_vicreg_pairwise_qwen25_3b.yaml \
            --output-dir runs/baseline/panoadapt_vicreg_pairwise_qwen25-vl-3b/eval
    fi
fi

# ----------------------------------------------------------
# 4. VICReg-pw 50% Gemma3-4B — TRAIN + EVAL
# ----------------------------------------------------------
echo ""
echo ">>> [4/4] VICReg-pw 50% Gemma3-4B train + eval"

$PYTHON -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml

with open('configs/baseline/panoadapt_vicreg_pairwise_gemma3_4b.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = BaselineConfig(**cfg_dict)
cfg.training.num_epochs = 0.001
trainer = BaselineTrainer(cfg)
trainer.train()
print('SMOKE TEST VICReg-pw 50% Gemma3 PASSED')
" 2>&1 | tee /tmp/smoke_vicreg_pw50_gemma3.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ SMOKE TEST FAILED — check /tmp/smoke_vicreg_pw50_gemma3.log"
    FAILED+=("[VICReg-pw 50% Gemma3] smoke test")
else
    echo "  ✅ SMOKE TEST PASSED"
    run_task "[VICReg-pw 50% Gemma3] training" \
        $PYTHON scripts/baseline_finetune.py \
        --config configs/baseline/panoadapt_vicreg_pairwise_gemma3_4b.yaml

    if [[ ! "${FAILED[*]}" =~ "VICReg-pw 50% Gemma3.*training" ]]; then
        run_task "[VICReg-pw 50% Gemma3] eval" \
            $PYTHON scripts/baseline_eval.py \
            --config configs/baseline/panoadapt_vicreg_pairwise_gemma3_4b.yaml \
            --output-dir runs/baseline/panoadapt_vicreg_pairwise_gemma3-4b/eval
    fi
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "=========================================="
echo "  GPU 1 Phase 2 — SUMMARY"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="
echo ""
echo "✅ Succeeded (${#SUCCEEDED[@]}):"
for s in "${SUCCEEDED[@]}"; do echo "   - $s"; done
echo ""
echo "❌ Failed (${#FAILED[@]}):"
for f in "${FAILED[@]}"; do echo "   - $f"; done
echo "=========================================="
