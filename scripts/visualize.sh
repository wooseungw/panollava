source /data/3_lib/miniconda3/etc/profile.d/conda.sh && conda activate pano
python scripts/visualize.py \
    --checkpoint runs/ADDDATA_SQ3_1_latent768_PE_anyres_erp_finetune_mlp/last.ckpt \
    --image data/quic360/downtest/images/2094501355_045ede6d89_k.jpg \
    --crop_strategy anyres_e2p