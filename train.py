# coding: utf-8
"""
Panorama-VLM Training with Resume/Warm-start
────────────────────────────────────────────
stage 선택
• vision      → model(stage="vicreg")   : Vision + VICReg
• resampler   → model(stage="train")    : Resampler 사전학습
• finetune    → model(stage="train")    : End-to-End SFT
"""
# ============================================================================
import argparse, torch, lightning as pl, wandb
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import default_data_collator
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# ── 내부 모듈 ---------------------------------------------------------------
from panovlm.processors.image          import PanoramaImageProcessor
from panovlm.processors.text           import TextTokenizer
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.dataset                   import ChatPanoDataset
from panovlm.model                     import PanoramaVLM
# ----------------------------------------------------------------------------

# =============================================================================
# 1. DataModule
# =============================================================================
class VLMDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, batch_size=4, num_workers=4,
                 image_size=(224,224), crop_strategy="e2p",
                 tokenizer_name="Qwen/Qwen3-0.6B", max_txt_len=2048):
        super().__init__()
        self.save_hyperparameters()
        img_proc = PanoramaImageProcessor(image_size=image_size,
                                          crop_strategy=crop_strategy)
        txt_tok  = TextTokenizer(tokenizer_name, max_len=max_txt_len,)
        self.processor = PanoLLaVAProcessor(img_proc, txt_tok)
        self.tokenizer = txt_tok.tok

    def setup(self, stage=None):
        self.train_ds = ChatPanoDataset(self.hparams.csv_train,
                                        self.processor, self.tokenizer)
        self.val_ds   = ChatPanoDataset(self.hparams.csv_val,
                                        self.processor, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers,
                          collate_fn=default_data_collator, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers,
                          collate_fn=default_data_collator, pin_memory=True)

# =============================================================================
# 2. LightningModule
# =============================================================================
class VLMModule(pl.LightningModule):
    _STAGE_MAP = {"vision":"vicreg", "resampler":"train", "finetune":"train"}

    def __init__(self, vision_name, lm_name, resampler, stage, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = PanoramaVLM(vision_name=vision_name, lm_name=lm_name,
                                 resampler=resampler)
        self._stage_key = self._STAGE_MAP[stage]

        # 파라미터 freeze
        # stage별로 학습 파라미터 설정
        if stage == "vision":
            # vision encoder만 학습
            for n, p in self.model.named_parameters():
                p.requires_grad = ("vision" in n)
        elif stage == "resampler":
            # vision encoder, resampler, lm_proj만 학습
            for n, p in self.model.named_parameters():
                if ("vision" in n) or ("resampler" in n) or ("lm_proj" in n):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif stage == "finetune":
            # lm을 제외한 모든 것 학습
            for n, p in self.model.named_parameters():
                if n.startswith("model.lm"):
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def forward(self, **batch):
        return self.model(stage=self._stage_key, **batch)

    def training_step(self, batch, _):
        out = self(**batch); self.log("loss", out["loss"], prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, _):
        out = self(**batch); self.log("val_loss", out["loss"], prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.hparams.lr, betas=(0.9,0.98), weight_decay=0.05
        )

# =============================================================================
# 3. 샘플 로깅 콜백
# =============================================================================
class LogSamplesCallback(pl.Callback):
    def __init__(self, tokenizer, num_samples=5, max_new_tokens=32):
        self.tok, self.n, self.m = tokenizer, num_samples, max_new_tokens
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        batch = next(iter(trainer.datamodule.val_dataloader()))
        batch = {k:v.to(pl_module.device) if torch.is_tensor(v) else v
                 for k,v in batch.items()}
        pixel = batch["pixel_values"][:self.n]
        out = pl_module.model(stage="generate", pixel_values=pixel,
                              max_new_tokens=self.m, temperature=0.7)
        preds = out["text"]
        tbl = wandb.Table(columns=["idx","image","pred"])
        for i in range(self.n):
            tbl.add_data(i, wandb.Image(pixel[i,0].cpu()), preds[i])
        trainer.logger.experiment.log({"val_samples":tbl}, commit=False)

# =============================================================================
# 4. main
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv-train", required=True)
    p.add_argument("--csv-val",   required=True)
    p.add_argument("--vision-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--lm-name",     default="Qwen/Qwen3-0.6B")
    p.add_argument("--resampler",   default="mlp")
    p.add_argument("--stage", choices=["vision","resampler","finetune"], default="vision")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-txt-len", type=int, default=2048)

    # Resume 옵션
    p.add_argument("--resume-from", default=None)
    p.add_argument("--warm-start", action="store_true")

    # W&B
    p.add_argument("--wandb-project", default="panorama-vlm")
    p.add_argument("--wandb-name",    default=None)
    args = p.parse_args()

    # ── DataModule ----------------------------------------------------------
    dm = VLMDataModule(args.csv_train, args.csv_val,
                       batch_size=args.batch_size, num_workers=args.num_workers,
                       tokenizer_name=args.lm_name, max_txt_len=args.max_txt_len)

    # ── 모델 생성 / 로드 ----------------------------------------------------
    if args.resume_from and not args.warm_start:
        lit_model = None  # Trainer.fit 에서 ckpt_path로 이어 학습
    else:
        if args.resume_from:  # Warm-start
            lit_model = VLMModule.load_from_checkpoint(
                args.resume_from,
                vision_name=args.vision_name, lm_name=args.lm_name,
                resampler=args.resampler, stage=args.stage, lr=args.lr,
                map_location="cpu",
            )
        else:                 # 새로 학습
            lit_model = VLMModule(args.vision_name, args.lm_name,
                                  args.resampler, args.stage, args.lr)

    # ── Logger & Callbacks --------------------------------------------------
    run_name = args.wandb_name or f"{args.stage}_{Path(args.csv_train).stem}"
    logger = WandbLogger(project=args.wandb_project, name=run_name,
                         config=vars(args))
    sample_cb = LogSamplesCallback(dm.tokenizer)
    ckpt_cb   = ModelCheckpoint(monitor="val_loss", mode="min",
                                save_top_k=1, filename="{epoch:02d}-{val_loss:.3f}")

    # ── Trainer -------------------------------------------------------------
    trainer = pl.Trainer(logger=logger, callbacks=[sample_cb, ckpt_cb],
                         max_epochs=args.epochs, precision=16,
                         gradient_clip_val=1.0, accelerator="auto",
                         default_root_dir=f"runs/vlm_{args.stage}")

    # ── Fit -----------------------------------------------------------------
    if args.resume_from and not args.warm_start:
        trainer.fit(lit_model, datamodule=dm, ckpt_path=args.resume_from)
    else:
        trainer.fit(lit_model, datamodule=dm)
