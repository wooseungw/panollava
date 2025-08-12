

python eval.py --ckpt runs/e2p_finetune_mlp/best.ckpt \
    --lora-weights-path runs/e2p_finetune_mlp/lora_weights \
    --csv-input data/quic360/test.csv \
    --batch-size 4 \
    --num-workers 4 \