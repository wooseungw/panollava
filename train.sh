
bash ./scripts/stage1_vision_train.sh 
bash ./scripts/stage2_resampler_train.sh ./runs/e2p_vision_mlp/vision/epoch=01-val_loss=1.209.ckpt 
bash ./scripts/stage3_finetune_train.sh ./runs/e2p_resampler_mlp/resampler/epoch=01-val_loss=1.297.ckpt
