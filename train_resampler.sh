export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
#!/bin/bash
# Resampler 스테이지 학습
python train.py \
    --stage resampler \
    --epochs 1 \
    --resume-from ./runs/vlm_vision/checkpoints/epoch=00-val_loss=1.234.ckpt \
    --csv-train "data/quic360/train.csv" \
    --csv-val "data/quic360/valid.csv" \
    --wandb-project "panorama-vlm"