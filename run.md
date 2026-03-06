PYTHON="/home/wsw/miniconda3/envs/pano/bin/python"
TRAIN="scripts/train.py"
EVAL="scripts/eval.py"
TEST_CSV="/data/1_personal/4_SWWOO/refer360/data/quic360_format/test.csv"
# ============================================================================
# 1. Loss Comparison Experiments
# ============================================================================
# ── C. InfoNCE — Stage 2,3 (현재 cora-train 세션에서 실행 중) ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $TRAIN \
  --config configs/cora/contrastive.yaml --resume auto \
  2>&1 | tee runs/cora_contrastive_stage2.log
# ── A. VICReg (batchwise) — Stage 2,3 (Stage 1 이미 완료, vicreg_batchwise.yaml은 stages: [resampler, finetune]) ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $TRAIN \
  --config configs/cora/vicreg_batchwise.yaml --resume auto \
  2>&1 | tee runs/cora_vicreg_batchwise.log
# ── B. VICReg (pairwise) — 전체 3-stage (Stage 1부터) ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $TRAIN \
  --config configs/cora/vicreg.yaml \
  2>&1 | tee runs/cora_vicreg.log
# ── D. DenseCL — 전체 3-stage (Stage 1부터) ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $TRAIN \
  --config configs/cora/densecl.yaml \
  2>&1 | tee runs/cora_densecl.log
# ============================================================================
# 2. Evaluation (각 실험 finetune 완료 후)
# ============================================================================
# ── C. InfoNCE eval ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $EVAL \
  --config configs/cora/contrastive.yaml \
  --checkpoint runs/cora_contrastive/*/finetune/last.ckpt \
  --test-csv $TEST_CSV \
  --output-dir outputs/cora_contrastive/
# ── A. VICReg (batchwise) eval ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $EVAL \
  --config configs/cora/vicreg_batchwise.yaml \
  --checkpoint runs/cora_vicreg_batchwise/*/finetune/last.ckpt \
  --test-csv $TEST_CSV \
  --output-dir outputs/cora_vicreg_batchwise/
# ── B. VICReg (pairwise) eval ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $EVAL \
  --config configs/cora/vicreg.yaml \
  --checkpoint runs/cora_vicreg/*/finetune/last.ckpt \
  --test-csv $TEST_CSV \
  --output-dir outputs/cora_vicreg/
# ── D. DenseCL eval ──
CUDA_VISIBLE_DEVICES=1 $PYTHON $EVAL \
  --config configs/cora/densecl.yaml \
  --checkpoint runs/cora_densecl/*/finetune/last.ckpt \
  --test-csv $TEST_CSV \
  --output-dir outputs/cora_densecl/
# ============================================================================
# 3. Baseline — InternVL3.5-1B 재평가 (빈 예측 버그 수정 후)
# ============================================================================
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/baseline_eval.py \
  --config configs/baseline/default.yaml \
  --test-csv $TEST_CSV \
  --output-dir runs/baseline/internvl3_5-1b_img256_tok128/