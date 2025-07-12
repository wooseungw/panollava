r"""
Panorama-VLM Multi-Stage Training Script
=======================================
• stage=vision          : Vision Encoder + Overlap Consistency (VICReg)
• stage=qformer        : Resampler + ITC/ITM/LM Pre-training
• stage=finetune       : Full End-to-End LM SFT

Example CLI
-----------
$ python train_panorama_vlm.py \
    --csv-train train.csv --csv-val val.csv --image-root /data/quic360 \
    --vision-name openai/clip-vit-base-patch32 \
    --lm-name Qwen/Qwen3-0.6B \
    --resampler conv \
    --stage vision --epochs 1 --batch-size 8
"""
# -------------------------------------------------------------
import os, argparse, json, math, random
from pathlib import Path

import torch, lightning as pl
from torch.utils.data import DataLoader
from transformers import default_data_collator

from panovlm.processors.image import PanoramaImageProcessor
from panovlm.processors.text import TextTokenizer
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.dataset import ChatPanoDataset
from panovlm.model import PanoramaVLM

# -------------------------------------------------------------
class VLMDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, image_root, batch_size=4, num_workers=4,
                 image_size=(224,224), crop_strategy="e2p", tokenizer="Qwen/Qwen3-0.6B"):
        super().__init__()
        self.save_hyperparameters()
        img_proc = PanoramaImageProcessor(image_size=image_size, crop_strategy=crop_strategy)
        txt_tok  = TextTokenizer(tokenizer)
        self.processor = PanoLLaVAProcessor(img_proc, txt_tok)
        self.tokenizer = txt_tok.tok

    def setup(self, stage=None):
        self.train_ds = ChatPanoDataset(self.hparams.csv_train, self.hparams.image_root,
                                        self.processor, self.tokenizer)
        self.val_ds   = ChatPanoDataset(self.hparams.csv_val,   self.hparams.image_root,
                                        self.processor, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, collate_fn=default_data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, collate_fn=default_data_collator)

# -------------------------------------------------------------
class VLMModule(pl.LightningModule):
    def __init__(self, vision_name, lm_name, resampler, stage, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = PanoramaVLM(vision_name=vision_name, lm_name=lm_name, resampler=resampler)
        self.stage = stage
        # parameter freezing rules
        if stage == "vision":
            for n,p in self.model.named_parameters():
                if "vision" in n: p.requires_grad = True
                else: p.requires_grad = False
        elif stage == "qformer":
            for p in self.model.vision.parameters(): p.requires_grad = False
        # finetune: all trainable

    def forward(self, **batch):
        return self.model(stage=self.stage, **batch)

    def training_step(self, batch, _):
        out = self(**batch)
        self.log("loss", out["loss"], prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, _):
        out = self(**batch)
        self.log("val_loss", out["loss"], prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.hparams.lr)

# -------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv-train"); p.add_argument("--csv-val"); p.add_argument("--image-root")
    p.add_argument("--vision-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--lm-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--resampler", default="identity")
    p.add_argument("--stage", choices=["vision","qformer","finetune"], default="vision")
    p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    dm = VLMDataModule(args.csv_train, args.csv_val, args.image_root,
                       batch_size=args.batch_size, num_workers=args.num_workers)
    model = VLMModule(args.vision_name, args.lm_name, args.resampler, args.stage, args.lr)

    trainer = pl.Trainer(max_epochs=args.epochs, precision=16, gradient_clip_val=1.0,
                         default_root_dir="runs/vlm_"+args.stage)
    trainer.fit(model, datamodule=dm)
