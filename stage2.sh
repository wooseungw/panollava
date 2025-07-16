export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

python train.py --resume-from runs/vlm_vision/checkpoints/epoch=00-val_loss=12.053.ckpt --stages vision resampler finetune \
  --epochs 1 \
  --lr 2e-6 \
  --batch-size 4 \
  --vicreg-loss-weight 0.5 \
  --max-txt-len 64 \
  --num-workers 16