

python eval.py --ckpt runs/dinov2qwen25_e2p_finetune_mlp/best.ckpt \
    --lora-weights-path runs/dinov2qwen25_e2p_finetune_mlp/lora_weights \
    --vision-name "facebook/dinov2-base" \
    --lm-name "Qwen/Qwen2.5-0.5B" \
    --resampler "mlp"\
    --crop-strategy "e2p" \
    --csv-input data/quic360/test.csv \
    --batch-size 4 \
    --num-workers 4 \
    --overlap-ratio 0.5 \