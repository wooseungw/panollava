

python eval.py --ckpt runs/siglipv2qwen25Instruct_e2p_finetune_mlp/best.ckpt \
    --lora-weights-path runs/siglipv2qwen25Instruct_e2p_finetune_mlp/lora_weights \
    --vision-name "google/siglip2-base-patch16-224" \
    --lm-name "Qwen/Qwen2.5-0.5B-Instruct" \
    --resampler "mlp"\
    --crop-strategy "e2p" \
    --csv-input data/quic360/test.csv \
    --batch-size 4 \
    --num-workers 4 \
    --overlap-ratio 0.5 \