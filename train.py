# coding: utf-8
"""
Panorama-VLM Training (Config-only)
───────────────────────────────────
- 단일 config.json 에서만 모든 설정을 읽음 (CLI 오버라이드 없음)
- stages:
    • "training.default_stage": 단일 스테이지 실행
    • "training.stages": ["vision","resampler","finetune"] 같이 여러 스테이지 순차 실행
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import json
import time
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
torch.set_float32_matmul_precision("high")

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner

# Plot 저장을 위한 matplotlib (선택적)
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 사용
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── 내부 모듈 ---------------------------------------------------------------
from panovlm.dataset   import VLMDataModule
from panovlm.model     import PanoramaVLM
from panovlm.utils     import *
from panovlm.config    import Config, ModelConfig, ConfigManager
# ----------------------------------------------------------------------------

# ── 로깅 설정 ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")]
)
logger = logging.getLogger("panovlm.train")

# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class VLMModule(pl.LightningModule):
    """Panorama VLM Lightning 래퍼 (stage-aware)"""
    _STAGE_MAP = {"vision": "vision", "resampler": "resampler", "finetune": "finetune", "generate": "generate"}

    def __init__(self, *, stage: str, model_config: ModelConfig, lr: float,
                 use_lora_cfg: Dict[str, Any], pretrained_dir: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])  # hparams에 덜어냄
        self.model_config: ModelConfig = model_config
        self.lr = lr  # 명시적으로 저장
        self.learning_rate = lr  # Lightning Tuner를 위한 속성

        # 모델 생성 우선순위: pretrained_dir > scratch
        if pretrained_dir and os.path.isdir(pretrained_dir):
            logger.info(f"🧩 Loading from pretrained dir: {pretrained_dir}")
            try:
                self.model = PanoramaVLM.from_pretrained_dir(
                    pretrained_dir,
                    **self.model_config.get_model_kwargs()
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load pretrained dir ({pretrained_dir}): {e}. Falling back to scratch init.")
                self.model = PanoramaVLM(**self.model_config.get_model_kwargs())
        else:
            self.model = PanoramaVLM(**self.model_config.get_model_kwargs())

        # stage 검증/매핑
        if stage not in self._STAGE_MAP:
            raise ValueError(f"stage는 {list(self._STAGE_MAP.keys())} 중 하나여야 합니다 (got: {stage})")
        self._stage_key = self._STAGE_MAP[stage]

        # LoRA 설정 (finetune에서만 적용)
        self.use_lora: bool = bool(use_lora_cfg.get("use_lora", False))
        if self.use_lora and stage == "finetune":
            logger.info("Setting up LoRA for finetune stage...")
            lora_kwargs = {
                "lora_r": use_lora_cfg.get("rank", 16),
                "lora_alpha": use_lora_cfg.get("alpha", 32),
                "lora_dropout": use_lora_cfg.get("dropout", 0.1),
                "target_modules": use_lora_cfg.get("target_modules", None),
            }
            ok = self.model.setup_lora_for_finetune(**lora_kwargs)
            if ok:
                logger.info(f"✓ LoRA setup completed: {lora_kwargs}")
            else:
                logger.warning("⚠ LoRA setup failed, continue with full finetune")
        elif self.use_lora and stage != "finetune":
            logger.warning(f"⚠ LoRA는 finetune 단계에서만 활성화됩니다. (현재: {stage}) → 무시")

        # stage별 동결/해제
        self._freeze_for_stage(stage)

        # 메타데이터(hparams)에 핵심 설정 저장
        self._prepare_checkpoint_metadata()

    # ── Lightning 표준 메서드들 ────────────────────────────────────────────
    def forward(self, **batch):
        return self.model(stage=self._stage_key, **batch)

    def training_step(self, batch, batch_idx):
        try:
            # 메모리 최적화: gradient checkpointing 활성화
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # 메모리 최적화: 중간 결과물들을 즉시 해제
            with torch.cuda.device(self.device) if torch.cuda.is_available() else torch.no_grad():
                out = self(**batch)
                loss = out["loss"]

            # 배치크기
            bs = None
            try:
                if isinstance(batch.get("pixel_values"), torch.Tensor):
                    bs = batch["pixel_values"].size(0)
            except Exception:
                pass

            # 수치 안정성
            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss at step {self.global_step}: {loss}")
                # 메모리 정리
                if 'out' in locals():
                    del out
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return None

            # 로깅
            kw = dict(prog_bar=True, sync_dist=True)
            if bs is not None: kw["batch_size"] = bs
            self.log("loss", loss, **kw)

            if "vicreg_loss" in out:
                self.log("train_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False, **({"batch_size": bs} if bs else {}))
            if "ar_loss" in out:
                self.log("train_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False, **({"batch_size": bs} if bs else {}))

            # wandb logger 추가 메트릭
            if self.trainer.logger is not None and batch_idx % 10 == 0:
                self.trainer.logger.log_metrics({
                    "train_loss": float(loss.detach().cpu()),
                    "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                    "global_step": self.global_step
                }, step=self.global_step)

            return loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM in training step {self.global_step}")
                # 적극적인 메모리 정리
                if 'out' in locals():
                    del out
                if 'loss' in locals():
                    del loss
                # 배치 데이터도 정리
                for key in list(batch.keys()):
                    if torch.is_tensor(batch[key]):
                        del batch[key]
                # GPU 메모리 완전히 비우기
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc
                gc.collect()
                # 메모리 상태 로깅
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                    logger.error(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                return None
            else:
                logger.error(f"Runtime error in training step {self.global_step}: {e}")
                # 일반 런타임 에러에도 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
        except Exception as e:
            logger.error(f"Unexpected error in training step {self.global_step}: {e}")
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
            return None

    def validation_step(self, batch, batch_idx):
        try:
            # 배치 정보 로깅 (첫 번째 배치에서만)
            if batch_idx == 0:
                logger.info(f"[VAL] First validation batch keys: {list(batch.keys())}")
                if "pixel_values" in batch:
                    logger.info(f"[VAL] pixel_values shape: {batch['pixel_values'].shape}")
            
            out = self(**batch)
            if "loss" not in out:
                logger.error(f"[VAL] No 'loss' key in model output. Keys: {list(out.keys())}")
                return None
                
            loss = out["loss"]

            bs = None
            try:
                if isinstance(batch.get("pixel_values"), torch.Tensor):
                    bs = batch["pixel_values"].size(0)
            except Exception:
                pass

            if not torch.isfinite(loss):
                logger.warning(f"[VAL] Non-finite val loss at step {batch_idx}: {loss}")
                return None

            # 메트릭 로깅 - on_epoch=True 추가로 epoch 레벨에서 집계되도록 함
            kw = dict(prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
            if bs is not None: kw["batch_size"] = bs
            self.log("val_loss", loss, **kw)

            # 추가 메트릭들도 on_epoch=True로 설정
            if "vicreg_loss" in out:
                kw_extra = dict(prog_bar=False, sync_dist=False, on_epoch=True, on_step=False)
                if bs is not None: kw_extra["batch_size"] = bs
                self.log("val_vicreg_loss", out["vicreg_loss"], **kw_extra)
            if "ar_loss" in out:
                kw_extra = dict(prog_bar=False, sync_dist=False, on_epoch=True, on_step=False)
                if bs is not None: kw_extra["batch_size"] = bs
                self.log("val_ar_loss", out["ar_loss"], **kw_extra)

            # 첫 번째 validation step에서 성공 메시지
            if batch_idx == 0:
                logger.info(f"[VAL] First validation step successful. Loss: {loss.item():.6f}")

            return loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[VAL] OOM in validation step {batch_idx}")
                # 메모리 정리
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                return None
            else:
                logger.error(f"[VAL] Runtime error in validation step {batch_idx}: {e}")
                return None
        except Exception as e:
            logger.error(f"[VAL] Error in validation step {batch_idx}: {e}")
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
            return None

    # ── 내부 유틸 ──────────────────────────────────────────────────────────
    def _freeze_for_stage(self, stage: str):
        self.model.requires_grad_(False)
        if stage == "vision":
            # 파노라마 적응을 위한 선택적 vision encoder 학습
            self.model.resampler.requires_grad_(True)
            self.model.vicreg_projector.requires_grad_(True)
            logger.info("✓ Stage 1: Selective vision layers + Resampler + VICReg projector unfrozen")
        elif stage == "resampler":
            # 더 많은 vision encoder 레이어 해제
            
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            logger.info("✓ Stage 2: Progressive vision layers + Resampler + Projection unfrozen")
        elif stage == "finetune":
            # 전체 vision encoder 미세조정 (낮은 학습률로)
            
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            if not self.use_lora:
                for p in self.model.language_model.parameters():
                    p.requires_grad = True
                logger.info("✓ Stage 3: Full model with adaptive learning rates")
            else:
                logger.info("✓ Stage 3: Vision + LoRA adapters with adaptive rates")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable:,}/{total:,} ({trainable/total:.1%})")


    def configure_optimizers(self):
        """파노라마 적응을 위한 차별화된 학습률 적용"""
        # 파라미터 그룹 분리
        vision_params = []
        resampler_params = []
        projection_params = []
        lm_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'vision_encoder' in name:
                vision_params.append(param)
            elif 'resampler' in name:
                resampler_params.append(param)
            elif 'vision_to_language_projection' in name or 'vicreg_projector' in name:
                projection_params.append(param)
            elif 'language_model' in name:
                lm_params.append(param)
            else:
                other_params.append(param)
        
        # 기본 학습률 (LR Finder가 업데이트할 수 있음)
        base_lr = getattr(self, 'learning_rate', self.hparams.lr)
        
        # 파라미터 그룹별 차별화된 학습률
        param_groups = []
        if vision_params:
            param_groups.append({
                'params': vision_params, 
                'lr': base_lr * 0.1,  # vision은 10배 낮은 학습률
                'weight_decay': 0.01
            })
        if resampler_params:
            param_groups.append({
                'params': resampler_params, 
                'lr': base_lr,  # 기본 학습률
                'weight_decay': 0.05
            })
        if projection_params:
            param_groups.append({
                'params': projection_params, 
                'lr': base_lr,  # 기본 학습률
                'weight_decay': 0.05
            })
        if lm_params:
            param_groups.append({
                'params': lm_params, 
                'lr': base_lr * 0.5,  # LM은 절반 학습률
                'weight_decay': 0.01
            })
        if other_params:
            param_groups.append({
                'params': other_params, 
                'lr': base_lr,
                'weight_decay': 0.05
            })
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)
        
        logger.info(f"Optimizer groups: Vision({len(vision_params)} params, lr={base_lr*0.1}), "
                   f"Resampler({len(resampler_params)} params, lr={base_lr}), "
                   f"Projection({len(projection_params)} params, lr={base_lr})")
        
        # 스케줄러
        try:
            from transformers import get_linear_schedule_with_warmup
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_steps = steps_per_epoch * self.trainer.max_epochs
            warmup_steps = max(1, int(0.1 * total_steps))
            sch = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            logger.info(f"✓ Scheduler: warmup {warmup_steps}, total {total_steps}")
            return [optimizer], [{"scheduler": sch, "interval": "step"}]
        except Exception as e:
            logger.warning(f"Scheduler init failed: {e}; Using optimizer only.")
            return optimizer

    def _prepare_checkpoint_metadata(self):
        meta = {
            "vision_name": self.model_config.vision_name,
            "language_model_name": self.model_config.language_model_name,
            "resampler_type": self.model_config.resampler_type,
            "latent_dimension": self.model_config.latent_dimension,
            "vicreg_loss_weight": self.model_config.vicreg_loss_weight,
            "vicreg_overlap_ratio": self.model_config.vicreg_overlap_ratio,
            "max_text_length": self.model_config.max_text_length,
            "stage": self._stage_key,
            "use_lora": self.use_lora
        }
        for k, v in meta.items():
            if k not in self.hparams:
                self.hparams[k] = v
        logger.info(f"✓ 체크포인트 메타데이터 준비 ({len(meta)} 항목)")

# ─────────────────────────────────────────────────────────────────────────────
# 콜백: 간단 모니터링
# ─────────────────────────────────────────────────────────────────────────────
class BatchSizeMonitorCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        logger.info("=== TRAIN START ===")
        # 모델 설정
        mc: ModelConfig = pl_module.model_config
        logger.info(f"[MODEL] vision={mc.vision_name} | lm={mc.language_model_name} | resampler={mc.resampler_type} | dim={mc.latent_dimension}")
        logger.info(f"[TEXT] max_len={mc.max_text_length} | LoRA={pl_module.use_lora}")
        # 데이터셋/로더
        logger.info(f"[DATA] train_csv={trainer.datamodule.hparams.csv_train}")
        logger.info(f"[DATA] val_csv={trainer.datamodule.hparams.csv_val}")
        logger.info(f"[DATA] image_size={trainer.datamodule.hparams.image_size} | crop={trainer.datamodule.hparams.crop_strategy}")
        # 로더 크기
        logger.info(f"[LOADER] train_batches={len(trainer.datamodule.train_dataloader())} | val_batches={len(trainer.datamodule.val_dataloader())}")
        # 환경
        if torch.cuda.is_available():
            logger.info(f"[GPU] count={torch.cuda.device_count()} | name={torch.cuda.get_device_name()}")

    def on_train_epoch_start(self, trainer, pl_module):
        _ = pl_module  # 사용하지 않는 매개변수 무시
        logger.info(f"[Epoch {trainer.current_epoch}] start")

    def on_train_epoch_end(self, trainer, pl_module):
        # 사용하지 않는 매개변수들을 명시적으로 무시
        _ = trainer, pl_module

# ─────────────────────────────────────────────────────────────────────────────
# 실행 유틸
# ─────────────────────────────────────────────────────────────────────────────
def load_config_dict() -> Dict[str, Any]:
    """config.json 로드 (환경변수 PANOVLM_CONFIG로 경로 지정 가능)"""
    cfg_path = os.environ.get("PANOVLM_CONFIG", "config.json")
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    logger.info(f"✓ Loaded config: {p}")
    return cfg

def get_stage_list(cfg: Dict[str, Any]):
    tr = cfg.get("training", {})
    stages = tr.get("stages", None)
    if isinstance(stages, list) and len(stages) > 0:
        return stages
    default_stage = tr.get("default_stage", None)
    if isinstance(default_stage, str):
        return [default_stage]
    # fallback
    return ["vision"]

def stage_defaults(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """코드 기본 + 파일 설정 병합 (파일 우선)"""
    code_default = Config.STAGE_DEFAULTS.get(stage, {})
    file_default = cfg.get("training", {}).get(stage, {})
    merged = {**code_default, **file_default}
    return merged

def build_datamodule(cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> VLMDataModule:
    data = cfg.get("data", {})
    paths = cfg.get("paths", {})
    ip  = cfg.get("image_processing", {})
    dm = VLMDataModule(
        csv_train=paths.get("csv_train") or data.get("csv_train"),
        csv_val=paths.get("csv_val") or data.get("csv_val"),
        batch_size=stage_cfg.get("batch_size", 1),  # Tuner가 최적 크기를 찾을 것
        num_workers=cfg.get("training", {}).get("num_workers", 16),
        image_size=tuple(ip.get("image_size", [224, 224])),
        tokenizer_name=(
            cfg.get("models", {}).get("language_model_name")
            or cfg.get("models", {}).get("lm_model", "Qwen/Qwen2.5-0.5B-Instruct")
        ),
        max_text_length=cfg.get("data", {}).get("max_text_length", 256),
        crop_strategy=ip.get("crop_strategy", "e2p"),
        system_msg=cfg.get("system_messages", {}).get("default", None),
        # Image processing extras
        overlap_ratio=ip.get("overlap_ratio", 0.5),
        fov_deg=ip.get("fov_deg", 90.0),
        image_mean=ip.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=ip.get("image_std", [0.229, 0.224, 0.225]),
        use_vision_processor=ip.get("use_vision_processor", False),
        vision_model_name=ip.get("vision_model_name", None),
        anyres_patch_size=ip.get("anyres_patch_size", 336),
        anyres_max_patches=ip.get("anyres_max_patches", 12),
        normalize=ip.get("normalize", True),
    )
    return dm

def build_model(cfg: Dict[str, Any], stage: str, stage_cfg: Dict[str, Any], pretrained_dir_override: Optional[str] = None) -> VLMModule:
    # ModelConfig: config.json을 평탄화하여 로딩
    model_config = ConfigManager.load_config(os.environ.get("PANOVLM_CONFIG", "config.json"))

    # 학습률/LoRA만 외부로
    lr = stage_cfg.get("lr", 2e-5)
    use_lora_cfg = cfg.get("lora", {})
    
    # 사전학습 디렉토리 (override > config)
    pretrained_dir = pretrained_dir_override or cfg.get("paths", {}).get("pretrained_dir")

    module = VLMModule(
        stage=stage,
        model_config=model_config,
        lr=lr,
        use_lora_cfg=use_lora_cfg,
        pretrained_dir=pretrained_dir
    )
    return module

def build_logger_and_callbacks(cfg: Dict[str, Any], stage: str, stage_cfg: Dict[str, Any], dm: VLMDataModule, lit_model: VLMModule):
    runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
    prefix   = cfg.get("training", {}).get("prefix", "panovlm")
    crop     = cfg.get("image_processing", {}).get("crop_strategy", "e2p")
    resampler= (
        cfg.get("models", {}).get("resampler_type")
        or cfg.get("models", {}).get("resampler", "mlp")
    )
    ckpt_dir = f"{runs_dir}/{prefix}_{crop}_{stage}_{resampler}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # wandb
    wandb_logger = None
    try:
        # 환경변수 세팅 (보안키는 환경에서)
        env = cfg.get("environment", {})
        if "cuda_visible_devices" in env:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(env["cuda_visible_devices"])
        if "wandb_project" in env:
            os.environ["WANDB_PROJECT"] = str(env["wandb_project"])

        # 기존 런 닫기
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

        run_name = f"{stage}_{Path(dm.hparams.csv_train).stem}_{int(time.time())}"
        wandb_config = {
            "stage": stage,
            "batch_size": dm.hparams.batch_size,
            "lr": lit_model.hparams.lr,
            "epochs": stage_cfg.get("epochs"),
            "csv_train": dm.hparams.csv_train,
            "csv_val": dm.hparams.csv_val,
            "image_size": dm.hparams.image_size,
            "crop_strategy": dm.hparams.crop_strategy,
            "system_msg": dm.hparams.system_msg
        }

        wandb_logger = WandbLogger(
            project=env.get("wandb_project", "panovlm"),
            name=run_name,
            config=wandb_config,
            dir="./runs",
            save_dir="./runs"
        )
    except Exception as e:
        logger.warning(f"WandB logger init failed: {e}; continue without WandB.")

    # callbacks
    callbacks = [BatchSizeMonitorCallback()]
    # EarlyStopping 다시 활성화 (메트릭 로깅 개선됨)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, mode="min", verbose=True, check_on_train_epoch_end=False
    )
    callbacks.append(early_stop)

    return wandb_logger, callbacks, ckpt_dir

def run_stage(cfg: Dict[str, Any], stage: str, prev_artifact_dir: Optional[str] = None) -> str:
    logger.info(f"=== RUN STAGE: {stage} ===")

    # stage defaults (파일이 코드 기본을 덮음)
    sdef = stage_defaults(cfg, stage)
    logger.info(f"[STAGE DEFAULTS] {sdef}")

    # 현재 스테이지 run 디렉토리
    runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
    prefix   = cfg.get("training", {}).get("prefix", "panovlm")
    crop     = cfg.get("image_processing", {}).get("crop_strategy", "e2p")
    resampler= (
        cfg.get("models", {}).get("resampler_type")
        or cfg.get("models", {}).get("resampler", "mlp")
    )
    ckpt_dir = f"{runs_dir}/{prefix}_{crop}_{stage}_{resampler}"
    # DataModule
    dm = build_datamodule(cfg, sdef)

    # 모델 (필요 시 이전 스테이지 safetensors 디렉토리로 초기화)
    lit_model = build_model(cfg, stage, sdef, pretrained_dir_override=prev_artifact_dir)

    # 로거/콜백
    wandb_logger, callbacks, ckpt_dir = build_logger_and_callbacks(cfg, stage, sdef, dm, lit_model)

    # Trainer - 메모리 최적화 설정 추가
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=1.0,
        max_epochs=sdef.get("epochs", 1),
        precision="16-mixed",
        gradient_clip_val=1.0,  # 더 큰 gradient clipping으로 안정성 향상
        accelerator="auto",
        default_root_dir=ckpt_dir,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True,
        # 메모리 최적화 설정들
        accumulate_grad_batches=sdef.get("accumulate_grad_batches", 2),
        
    )

    # 메모리 상태 로깅
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"📊 GPU Memory after tuning - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    # 학습
    try:
        logger.info(f"Starting training (stage={stage})")
        t0 = time.time()
        trainer.fit(lit_model, datamodule=dm)
        logger.info(f"Training finished in {(time.time()-t0)/60:.1f} min")
    except Exception as e:
        logger.error(f"Training failed (stage={stage}): {e}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 모델 저장 (LoRA only 저장 옵션은 config의 lora.save_lora_only를 따름)
    save_lora_only = bool(cfg.get("lora", {}).get("save_lora_only", False))
    skip_full_save = (stage == "finetune" and lit_model.use_lora and save_lora_only)

    if not skip_full_save:
        try:
            logger.info("💾 Saving HF-style safetensors artifact...")
            hf_dir = str(Path(ckpt_dir) / "hf_model")
            lit_model.model.save_pretrained(hf_dir)
            logger.info(f"✓ HF style saved: {hf_dir}")
            # 간편 로딩 디렉토리 별칭
            simp = Path(ckpt_dir) / "panorama_model"
            if simp.exists():
                import shutil; shutil.rmtree(simp)
            import shutil; shutil.copytree(hf_dir, str(simp))
            logger.info(f"✓ Simple load dir: {simp}")
        except Exception as e:
            logger.error(f"❌ Model save failed: {e}")
    else:
        logger.info("💾 Skipping full save (LoRA-only enabled)")

    # LoRA 가중치 별도 저장
    if stage == "finetune" and lit_model.use_lora:
        try:
            lora_dir = str(Path(ckpt_dir) / "lora_weights")
            lit_model.model.save_lora_weights(lora_dir)
            logger.info(f"✓ LoRA weights saved to: {lora_dir}")
        except Exception as e:
            logger.warning(f"LoRA save failed: {e}")

    # 사용 안내
    logger.info("=" * 80)
    logger.info("🎉 훈련 완료! 모델 사용법:")
    logger.info("=" * 80)
    if Path(ckpt_dir, "hf_model").exists():
        logger.info(f"- HF model dir: {Path(ckpt_dir, 'hf_model')}")
    if Path(ckpt_dir, "panorama_model").exists():
        logger.info(f"- Simple load dir: {Path(ckpt_dir, 'panorama_model')}")
    if Path(ckpt_dir, "lora_weights").exists():
        logger.info(f"- LoRA weights: {Path(ckpt_dir, 'lora_weights')}")

    # 다음 스테이지를 위해 panorama_model 디렉토리 경로 반환
    return str(Path(ckpt_dir) / "panorama_model")

def run_all(cfg: Dict[str, Any]):
    # 환경변수 적용(옵션)
    env = cfg.get("environment", {})
    if "cuda_visible_devices" in env:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env["cuda_visible_devices"])
    if "wandb_project" in env:
        os.environ["WANDB_PROJECT"] = str(env["wandb_project"])
    # wandb_api_key는 반드시 외부 환경변수로만 주입하세요.

    stages = get_stage_list(cfg)
    logger.info(f"Planned stages: {stages}")

    prev_artifact = None
    for st in stages:
        prev_artifact = run_stage(cfg, st, prev_artifact_dir=prev_artifact)

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config_dict()
    run_all(cfg)
