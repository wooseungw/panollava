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
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
torch.set_float32_matmul_precision('high')  # H100 등에서 Tensor Core 최적화
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
                 tokenizer_name="Qwen/Qwen3-0.6B", max_txt_len=512):  # 기본값을 512로 줄임
        super().__init__()
        self.save_hyperparameters()
        img_proc = PanoramaImageProcessor(image_size=image_size,
                                          crop_strategy=crop_strategy)
        txt_tok  = TextTokenizer(tokenizer_name, max_len=max_txt_len,)
        self.processor = PanoLLaVAProcessor(img_proc, txt_tok, max_length=512)  # 최대 길이 512로 제한
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
    # stage 매핑: 내부적으로 허용되는 값으로 변환
    _STAGE_MAP = {"vision": "vision", "resampler": "finetune", "finetune": "finetune", "generate": "generate"}

    def __init__(self, vision_name, lm_name, resampler, stage, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = PanoramaVLM(
            vision_model_name=vision_name,
            language_model_name=lm_name,
            resampler_type=resampler
        )
        # stage가 허용되지 않은 값이면 에러
        if stage not in self._STAGE_MAP:
            raise ValueError(f"stage는 {list(self._STAGE_MAP.keys())} 중 하나여야 합니다")
        self._stage_key = self._STAGE_MAP[stage]
        self._freeze_for_stage(stage)

    def _freeze_for_stage(self, stage):
        # 전부 잠그고
        self.model.requires_grad_(False)

        if stage == "vision":
            self.model.vision_encoder.requires_grad_(True)

        elif stage == "resampler":
            self.model.vision_encoder.requires_grad_(True)
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)

        elif stage == "finetune":
            # LM 제외 전체
            # self.model.vision_encoder.requires_grad_(True)
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)

    def forward(self, **batch):
        return self.model(stage=self._stage_key, **batch)


    def training_step(self, batch, _):
        out = self(**batch)
        # loss만 로깅
        self.log("loss", out["loss"], prog_bar=True)
        # train 로그 파일에 loss 기록
        if self.trainer.logger is not None:
            self.trainer.logger.log_metrics({"train_loss": out["loss"].item()}, step=self.global_step)
        return out["loss"]

    def validation_step(self, batch, _):
        out = self(**batch)
        self.log("val_loss", out["loss"], prog_bar=True)
        if "vicreg_loss" in out:
            self.log("val_vicreg_loss", out["vicreg_loss"], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.hparams.lr, betas=(0.9,0.98), weight_decay=0.05
        )
        # Warmup 스케줄러 추가 (예: 10% warmup)
        from transformers import get_linear_schedule_with_warmup
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = steps_per_epoch * self.trainer.max_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# =============================================================================
# 3. 샘플 로깅 콜백
# =============================================================================
class LogSamplesCallback(pl.Callback):
    def __init__(self, tokenizer, num_samples=5, max_new_tokens=32):
        self.tok, self.n, self.m = tokenizer, num_samples, max_new_tokens
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        batch = next(iter(trainer.datamodule.val_dataloader()))
        batch = {k: v.to(pl_module.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pixel = batch["pixel_values"]
        input_ids = batch.get("input_ids", None)
        image_paths = batch.get("image_path", None)
        actual_n = min(self.n, pixel.shape[0])
        if actual_n == 0:
            print("[LogSamplesCallback] Warning: validation batch is empty, skipping sample logging.")
            return
        out = pl_module.model(stage="generate", pixel_values=pixel[:actual_n], max_new_tokens=self.m, temperature=0.7)
        preds = out["text"]
        # input_ids를 string으로 디코딩
        input_texts = None
        if input_ids is not None:
            # input_ids shape: (batch, seq_len)
            input_texts = self.tok.batch_decode(input_ids[:actual_n], skip_special_tokens=True)
        if len(preds) < actual_n:
            print(f"[LogSamplesCallback] Warning: model returned fewer predictions ({len(preds)}) than requested ({actual_n}).")
        tbl = wandb.Table(columns=["idx", "image", "image_path", "input_text", "pred"])
        for i in range(actual_n):
            # pixel[i] shape: (3, H, W) or (B, 3, H, W)?
            img = pixel[i]
            if img.dim() == 4:
                img = img[0]  # (B, 3, H, W) -> (3, H, W)
            input_str = input_texts[i] if input_texts is not None else "<no input>"
            pred_str = preds[i] if i < len(preds) else "<no prediction>"
            img_path = image_paths[i] if image_paths is not None else "<no path>"
            tbl.add_data(i, wandb.Image(img.cpu()), img_path, input_str, pred_str)
        trainer.logger.experiment.log({"val_samples": tbl}, commit=False)

# =============================================================================
# 4. main
# =============================================================================

# 단일 스테이지 학습
def run_single_stage(args, stage, prev_ckpt=None):
    return _run_stage_core(args, stage, prev_ckpt)

# 전체 스테이지 반복 학습
def run_all_stages(args, stages, prev_ckpt=None):
    for stage in stages:
        args.stage = stage
        print(f"\n===== [STAGE: {stage}] =====")
        prev_ckpt = _run_stage_core(args, stage, prev_ckpt)
        print(f"[STAGE: {stage}] best checkpoint: {prev_ckpt}")
    return prev_ckpt

# 내부 공통 학습 함수
def _run_stage_core(args, stage, prev_ckpt=None):
    # 스테이지별 기본값, 외부 인자가 없을 때만 적용
    stage_hparams = {
        "vision":    {"epochs": 1, "lr": 5e-6, "batch_size": 32, "vicreg_loss_weight": 1.0, "max-txt-len": 32},
        "resampler": {"epochs": 1, "lr": 2e-6, "batch_size": 16, "vicreg_loss_weight": 0.0, "max-txt-len": 64},
        "finetune":  {"epochs": 1, "lr": 2e-6, "batch_size": 16, "vicreg_loss_weight": 0.0, "max-txt-len": 128},
    }[stage]
    # argparse 기본값과 구분: None 또는 0/0.0/빈값이면 스테이지별 기본값 사용
    for k, v in stage_hparams.items():
        cur = getattr(args, k, None)
        # epochs/batch_size는 0 또는 None일 때, lr/vicreg_loss_weight는 0.0 또는 None일 때만 기본값 적용
        if cur is None or (isinstance(v, int) and (cur == 0)) or (isinstance(v, float) and (cur == 0.0)):
            setattr(args, k, v)

    dm = VLMDataModule(args.csv_train, args.csv_val,
                       batch_size=args.batch_size, num_workers=args.num_workers,
                       tokenizer_name=args.lm_name, max_txt_len=args.max_txt_len)
    is_stage_change = False
    if prev_ckpt:
        try:
            ckpt_hparams = torch.load(prev_ckpt, map_location="cpu").get('hyper_parameters', {})
            prev_stage = ckpt_hparams.get('stage')
            if prev_stage and prev_stage != stage:
                is_stage_change = True
                print(f"[INFO] Stage changed ({prev_stage} → {stage}): Loading weights only.")
        except Exception as e:
            print(f"[WARN] Could not read stage from checkpoint hparams: {e}. Assuming stage change.")
            is_stage_change = True

    if prev_ckpt:
        lit_model = VLMModule(args.vision_name, args.lm_name, args.resampler, stage, args.lr)
        if str(prev_ckpt).endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(prev_ckpt)
                print(f"[INFO] Loaded weights from safetensors: {prev_ckpt}")
            except ImportError:
                print("[WARN] safetensors 패키지가 설치되어 있지 않습니다. pip install safetensors 후 사용 가능합니다.")
                state_dict = None
            except Exception as e:
                print(f"[WARN] safetensors 로딩 중 오류: {e}")
                state_dict = None
        else:
            ckpt = torch.load(prev_ckpt, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            print(f"[INFO] Loaded weights from checkpoint: {prev_ckpt}")
        if state_dict is not None:
            missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded weights. Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        lit_model = VLMModule(args.vision_name, args.lm_name, args.resampler, stage, args.lr)

    run_name = f"{stage}_{Path(args.csv_train).stem}"
    wandb_dir = "./runs"
    wandb_config = {
        "stage": stage,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "vicreg_loss_weight": getattr(args, "vicreg_loss_weight", None),
        "vision_name": args.vision_name,
        "lm_name": args.lm_name,
        "resampler": args.resampler,
        "max_txt_len": args.max_txt_len,
        "csv_train": args.csv_train,
        "csv_val": args.csv_val,
        "num_workers": args.num_workers,
    }
    logger = WandbLogger(project=args.wandb_project, name=run_name, config=wandb_config, dir=wandb_dir)
    sample_cb = LogSamplesCallback(dm.tokenizer)
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        filename="{epoch:02d}-{val_loss:.3f}",
        dirpath=f"./runs/vlm_{stage}/checkpoints"
    )
    trainer = pl.Trainer(
        logger=logger, callbacks=[sample_cb, ckpt_cb],
        max_epochs=args.epochs, precision="16-mixed",
        gradient_clip_val=0.5, accelerator="auto",
        default_root_dir=f"./runs/vlm_{stage}"
    )

    if prev_ckpt and is_stage_change:
        trainer.fit(lit_model, datamodule=dm)
    else:
        trainer.fit(lit_model, datamodule=dm, ckpt_path=None)

    try:
        from safetensors.torch import save_file as save_safetensors
        safetensor_path = f"./runs/vlm_{stage}/model_final.safetensors"
        save_safetensors(lit_model.state_dict(), safetensor_path)
        print(f"[INFO] Model weights saved as safetensors: {safetensor_path}")
    except ImportError:
        print("[WARN] safetensors 패키지가 설치되어 있지 않습니다. pip install safetensors 후 사용 가능합니다.")
    except Exception as e:
        print(f"[WARN] safetensors 저장 중 오류: {e}")
    return ckpt_cb.best_model_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv-train", default="data/quic360/train.csv")
    p.add_argument("--csv-val", default="data/quic360/valid.csv")
    p.add_argument("--vision-name", default="google/siglip-base-patch16-224")
    p.add_argument("--lm-name",     default="Qwen/Qwen3-0.6B")
    p.add_argument("--resampler",   default="mlp")
    p.add_argument("--stage", choices=["vision","resampler","finetune"], default="vision")
    p.add_argument("--stages", nargs="*", default=None,
                   help="학습할 스테이지 리스트 (예: vision resampler finetune)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--vicreg-loss-weight", type=float, default=0.0, help="VICReg loss weight for each stage")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-txt-len", type=int, default=128)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--wandb-project", default="panorama-vlm")
    p.add_argument("--wandb-name",    default=None)
    args = p.parse_args()

    # 전체 스테이지 반복 학습 (--stages 지정 시)
    if args.stages is not None and len(args.stages) > 0:
        stages = args.stages if isinstance(args.stages, list) else args.stages.split()
        prev_ckpt = args.resume_from if args.resume_from else None
        run_all_stages(args, stages, prev_ckpt=prev_ckpt)
    else:
        # 단일 스테이지만 학습
        print(f"\n===== [SINGLE STAGE: {args.stage}] =====")
        prev_ckpt = run_single_stage(args, args.stage, prev_ckpt=args.resume_from if args.resume_from else None)
        print(f"[STAGE: {args.stage}] best checkpoint: {prev_ckpt}")
