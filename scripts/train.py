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
import argparse
# Silence HF tokenizers fork/parallelism warnings and avoid deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
from dataclasses import dataclass

import torch
torch.set_float32_matmul_precision("high")

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency for YAML configs
    yaml = None

# Plot 저장을 위한 matplotlib (선택적)
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 사용
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── 내부 모듈 ---------------------------------------------------------------
from panovlm.data     import VLMDataModule
from panovlm.models   import PanoramaVLM
from panovlm.utils    import *
from panovlm.config   import ConfigManager
from panovlm.config import ModelConfig

# 시각화 모듈 추가
try:
    from panovlm.evaluation.dino import DinoVisualizer
    DINO_AVAILABLE = True
except ImportError:
    DinoVisualizer = None
    DINO_AVAILABLE = False
# ----------------------------------------------------------------------------

# ── 로깅 설정 ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")]
)
logger = logging.getLogger("panovlm.train")

# 시각화 모듈 로드 확인
if not DINO_AVAILABLE:
    logger.warning("DinoVisualizer를 로드할 수 없습니다. 시각화가 비활성화됩니다.")

# Stage alias map keeps CLI/config names in sync with model internals
_STAGE_ALIAS_MAP = {
    "vision": "vision",
    "vision_pretraining": "vision",
    "vision_pretrain": "vision",
    "resampler": "resampler",
    "resampler_training": "resampler",
    "finetune": "finetune",
    "generate": "generate",
    "inference": "generate",
}


def _canonical_stage_name(stage: str) -> str:
    key = str(stage).strip()
    canonical = _STAGE_ALIAS_MAP.get(key)
    if canonical is None:
        valid = ", ".join(sorted(_STAGE_ALIAS_MAP.keys()))
        raise ValueError(f"stage는 [{valid}] 중 하나여야 합니다 (got: {stage})")
    return canonical


# ─────────────────────────────────────────────────────────────────────────────
# 시각화 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────
def _run_stage_visualization(cfg: Dict[str, Any], stage: str, checkpoint_path: Optional[str] = None) -> None:
    """각 단계 완료 후 DINO 시각화를 실행합니다."""
    if not DINO_AVAILABLE:
        logger.info(f"시각화 건너뜀: DinoVisualizer를 사용할 수 없습니다.")
        return

    try:
        logger.info(f"📊 {stage} 단계 시각화 시작...")

        # 시각화 저장 디렉토리 생성 (설정에서 runs_dir 사용)
        runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
        results_dir = Path("results")  # 시각화는 results 폴더에 저장
        vis_dir = results_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 현재 단계의 시각화 파일명 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        vis_prefix = f"{stage}_{timestamp}"

        # 간단한 시각화 실행 (실제로는 모델에서 hidden states를 추출해야 함)
        # 여기서는 예시로 빈 시각화 생성
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'{stage.upper()} 단계 완료\n{timestamp}',
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 시각화 저장
        vis_path = vis_dir / f"{vis_prefix}_completion.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ {stage} 단계 시각화 저장됨: {vis_path}")

        # 추가: 단계별 메트릭 요약 (실제로는 모델 평가 결과 사용)
        metrics_path = vis_dir / f"{vis_prefix}_metrics.json"
        metrics = {
            "stage": stage,
            "timestamp": timestamp,
            "checkpoint": checkpoint_path,
            "status": "completed"
        }

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"📈 메트릭 저장됨: {metrics_path}")

    except Exception as e:
        logger.warning(f"⚠️ {stage} 단계 시각화 실패: {e}")
        # 시각화 실패해도 학습은 계속 진행

# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class VLMModule(pl.LightningModule):
    """Panorama VLM Lightning 래퍼 (stage-aware)"""

    def __init__(self, *, stage: str, model_config: ModelConfig, lr: float,
                 use_lora_cfg: Dict[str, Any], pretrained_dir: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])  # hparams에 덜어냄
        self.model_config: ModelConfig = model_config
        self.lr = lr  # 명시적으로 저장
        self.learning_rate = lr  # Lightning Tuner를 위한 속성

        # 모델 생성 우선순위: pretrained_dir(.ckpt 또는 HF 디렉토리) > scratch
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
        elif pretrained_dir and os.path.isfile(pretrained_dir) and str(pretrained_dir).endswith('.ckpt'):
            logger.info(f"🧩 Loading from checkpoint file: {pretrained_dir}")
            try:
                self.model = PanoramaVLM.from_checkpoint(
                    pretrained_dir,
                    **self.model_config.get_model_kwargs()
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load checkpoint file ({pretrained_dir}): {e}. Falling back to scratch init.")
                self.model = PanoramaVLM(**self.model_config.get_model_kwargs())
        else:
            self.model = PanoramaVLM(**self.model_config.get_model_kwargs())

        # stage 검증/매핑
        try:
            self._stage_key = _canonical_stage_name(stage)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if stage != self._stage_key:
            logger.info(f"Stage alias resolved: '{stage}' → '{self._stage_key}'")

        # LoRA 설정 (finetune에서 적용)
        self.use_lora: bool = bool(use_lora_cfg.get("use_lora", False))
        if self.use_lora and self._stage_key == "finetune":
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
        elif self.use_lora and self._stage_key != "finetune":
            logger.warning(f"⚠ LoRA는 finetune 단계에서만 활성화됩니다. (현재: {stage}) → 무시")

        # stage별 동결/해제
        self._unfreeze_for_stage(self._stage_key)

        # 메타데이터(hparams)에 핵심 설정 저장
        self._prepare_checkpoint_metadata()

    # ── Lightning 표준 메서드들 ────────────────────────────────────────────
    def forward(self, **batch):
        return self.model(stage=self._stage_key, **batch)
    # VLMModule 내부: gradient checkpointing은 fit 시작 시 1회만 활성화
    def on_fit_start(self) -> None:
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("✓ Gradient checkpointing enabled (once on fit start)")
            except Exception as e:
                logger.warning(f"⚠️ Gradient checkpointing enable failed: {e}")

    # VLMModule 내부: OOM 시 모든 랭크에서 '0-loss'를 반환해 스텝 대칭 유지 (None 금지)
    def training_step(self, batch, batch_idx):
        try:
            out = self(**batch)
            loss = out["loss"]

            bs = None
            try:
                if isinstance(batch.get("pixel_values"), torch.Tensor):
                    bs = batch["pixel_values"].size(0)
            except Exception:
                pass

            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss at step {self.global_step}: {loss}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 대칭 스텝 처리
                return torch.zeros([], device=self.device, requires_grad=True)

            kw = dict(prog_bar=True, sync_dist=True)
            if bs is not None:
                kw["batch_size"] = bs
            self.log("loss", loss, **kw)

            if "vicreg_loss" in out:
                self.log("train_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False, **({"batch_size": bs} if bs else {}))
            if "ar_loss" in out:
                self.log("train_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False, **({"batch_size": bs} if bs else {}))

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
                # 적극적 정리
                try:
                    for k in list(batch.keys()):
                        if torch.is_tensor(batch[k]):
                            del batch[k]
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                # 모든 랭크 동기화 후 0-loss 반환(스텝 대칭)
                try:
                    self.trainer.strategy.barrier()
                except Exception:
                    pass
                return torch.zeros([], device=self.device, requires_grad=True)
            else:
                logger.error(f"Runtime error in training step {self.global_step}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 스텝 대칭 유지
                return torch.zeros([], device=self.device, requires_grad=True)
        except Exception as e:
            logger.error(f"Unexpected error in training step {self.global_step}: {e}")
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
            # 스텝 대칭 유지
            return torch.zeros([], device=self.device, requires_grad=True)

    # VLMModule 내부: 검증도 동일하게 None 금지(대칭 스텝 유지)
    def validation_step(self, batch, batch_idx):
        try:
            if batch_idx == 0:
                logger.info(f"[VAL] First validation batch keys: {list(batch.keys())}")
                if "pixel_values" in batch:
                    logger.info(f"[VAL] pixel_values shape: {batch['pixel_values'].shape}")

            out = self(**batch)
            if "loss" not in out:
                logger.error(f"[VAL] No 'loss' key in model output. Keys: {list(out.keys())}")
                return torch.zeros([], device=self.device)  # 대칭 리턴

            loss = out["loss"]

            if not torch.isfinite(loss):
                logger.warning(f"[VAL] Non-finite val loss at step {batch_idx}: {loss}")
                return torch.zeros([], device=self.device)  # 대칭 리턴

            kw = dict(prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
            self.log("val_loss", loss, **kw)

            if "vicreg_loss" in out:
                self.log("val_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False, on_epoch=True, on_step=False)
            if "ar_loss" in out:
                self.log("val_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False, on_epoch=True, on_step=False)

            return loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[VAL] OOM in validation step {batch_idx}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                try:
                    self.trainer.strategy.barrier()
                except Exception:
                    pass
                return torch.zeros([], device=self.device)  # 대칭 리턴
            else:
                logger.error(f"[VAL] Runtime error in validation step {batch_idx}: {e}")
                return torch.zeros([], device=self.device)  # 대칭 리턴
        except Exception as e:
            logger.error(f"[VAL] Error in validation step {batch_idx}: {e}")
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
            return torch.zeros([], device=self.device)  # 대칭 리턴

    def on_validation_epoch_end(self) -> None:
        try:
            if self.trainer is None:
                return
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                try:
                    val_loss_value = float(val_loss)
                except (TypeError, ValueError):
                    val_loss_value = val_loss
                logger.info(f"[VAL][Epoch {self.current_epoch}] mean loss: {val_loss_value:.6f}" if isinstance(val_loss_value, float)
                            else f"[VAL][Epoch {self.current_epoch}] mean loss: {val_loss_value}")
        except Exception as e:
            logger.warning(f"[VAL] Failed to log epoch summary: {e}")

    # ── 내부 유틸 ──────────────────────────────────────────────────────────
    def _unfreeze_for_stage(self, stage: str):
        # 모든 파라미터를 freeze한 뒤, 필요한 부분만 unfreeze
        self.model.requires_grad_(False)
        if stage == "vision":
            # Vision encoder와 VICReg projector 학습 (resampler/LM은 freeze 유지)
            if hasattr(self.model, "resampler"):
                self.model.resampler.requires_grad_(True)
            if hasattr(self.model, "vicreg_projector"):
                self.model.vicreg_projector.requires_grad_(True)
            logger.info("✓ Stage 1: Vision backbone + VICReg projector unfrozen")
        elif stage == "resampler":
            # 더 많은 vision encoder 레이어와 resampler, projection unfreeze
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            logger.info("✓ Stage 2: Progressive vision layers + Resampler + Projection unfrozen")
        elif stage == "finetune":
            # 전체 모델 unfreeze (단, LM은 항상 freeze)
            # if hasattr(self.model, "vision_backbone"):
            #     self.model.vision_backbone.requires_grad_(True)
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            # LM은 항상 freeze (LoRA 사용 여부와 무관)
            for p in self.model.language_model.parameters():
                p.requires_grad = False
            logger.info("✓ Stage 3 (Fine-tuning with Instruction Tuning): Vision + Resampler + Projection unfrozen (LM frozen, LoRA only if enabled)")

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
                'lr': base_lr,  # vision은 10배 낮은 학습률
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
                'lr': base_lr,  # LM은 절반 학습률
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
            "overlap_ratio": self.model_config.overlap_ratio,
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

def _available_stage_names(cfg: Dict[str, Any]) -> list[str]:
    training_cfg = cfg.get("training", {}) or {}
    stages = training_cfg.get("stages")
    if isinstance(stages, list) and stages:
        return stages
    default_stage = training_cfg.get("default_stage")
    if isinstance(default_stage, str) and default_stage:
        return [default_stage]
    return ["vision"]


def _normalize_dataset_dict(dataset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(dataset, dict):
        return {}
    normalized = dict(dataset)
    if "train_csv" in normalized and "csv_train" not in normalized:
        normalized["csv_train"] = normalized["train_csv"]
    if "val_csv" in normalized and "csv_val" not in normalized:
        normalized["csv_val"] = normalized["val_csv"]
    if "train" not in normalized and isinstance(normalized.get("csv_train"), list):
        normalized["train"] = normalized.get("csv_train")
    if "val" not in normalized and isinstance(normalized.get("csv_val"), list):
        normalized["val"] = normalized.get("csv_val")
    return normalized


def _convert_yaml_config(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    experiment = yaml_cfg.get("experiment")
    if isinstance(experiment, dict):
        result["experiment"] = experiment

    model_block = yaml_cfg.get("model") or {}
    vision_cfg = model_block.get("vision") or {}
    language_cfg = model_block.get("language") or {}
    resampler_cfg = model_block.get("resampler") or {}

    models = {
        "vision_model_name": vision_cfg.get("name"),
        "vision_name": vision_cfg.get("name"),
        "language_model_name": language_cfg.get("name"),
        "lm_model": language_cfg.get("name"),
        "resampler_type": resampler_cfg.get("type", "mlp"),
        "resampler": resampler_cfg.get("type", "mlp"),
        "latent_dimension": resampler_cfg.get("latent_dimension", 768),
        "resampler_depth": resampler_cfg.get("depth", 2),
        "resampler_hidden_dim": resampler_cfg.get("hidden_dim"),
        "resampler_use_ln": resampler_cfg.get("use_ln", True),
        "resampler_enable_cross_view": resampler_cfg.get("enable_cross_view", False),
        "resampler_num_views": resampler_cfg.get("num_views", 8),
        "resampler_dropout": resampler_cfg.get("dropout", 0.1),
        "resampler_heads": resampler_cfg.get("heads", 8),
        "resampler_num_latents": resampler_cfg.get("num_latents", 32),
    }
    result["models"] = {k: v for k, v in models.items() if v is not None}

    lora_cfg = model_block.get("lora")
    if isinstance(lora_cfg, dict):
        result["lora"] = {
            "use_lora": bool(lora_cfg.get("enabled", False)),
            "rank": lora_cfg.get("rank", 16),
            "alpha": lora_cfg.get("alpha", 32),
            "dropout": lora_cfg.get("dropout", 0.1),
            "target_modules": lora_cfg.get("target_modules"),
        }
    else:
        result["lora"] = {"use_lora": False}

    data_block = yaml_cfg.get("data") or {}
    default_dataset = _normalize_dataset_dict(data_block.get("default_dataset"))
    result["data"] = {k: v for k, v in default_dataset.items() if v is not None}

    image_processing = yaml_cfg.get("image_processing") or {}
    if isinstance(image_processing, dict):
        result["image_processing"] = dict(image_processing)

    env_cfg = yaml_cfg.get("environment") or {}
    if isinstance(env_cfg, dict):
        result["environment"] = dict(env_cfg)

    for key in ("evaluation", "hardware"):
        if isinstance(yaml_cfg.get(key), dict):
            result[key] = dict(yaml_cfg[key])

    training_block = yaml_cfg.get("training") or {}
    global_training = training_block.get("global") or {}

    stage_order = training_block.get("stage_order") or []
    stages_block = training_block.get("stages") or {}
    stage_configs: Dict[str, Any] = {}
    for stage_name, stage_cfg in stages_block.items():
        if not isinstance(stage_cfg, dict):
            continue
        converted = dict(stage_cfg)
        lr_value = converted.pop("learning_rate", converted.get("lr"))
        if lr_value is not None:
            converted["lr"] = lr_value
        if "data" in converted:
            converted["data"] = _normalize_dataset_dict(converted.get("data"))
        if isinstance(converted.get("image_processing"), dict):
            converted["image_processing"] = dict(converted["image_processing"])
        stage_configs[stage_name] = converted

    if not stage_order:
        stage_order = list(stage_configs.keys())

    training_out: Dict[str, Any] = {
        "stages": stage_order,
        "stage_configs": stage_configs,
    }

    if experiment and experiment.get("name"):
        training_out.setdefault("prefix", experiment.get("name"))

    for key in ("num_workers", "wandb_project", "system_msg", "empty_cache_each_step"):
        if key in global_training and global_training[key] is not None:
            training_out[key] = global_training[key]

    if isinstance(training_block.get("deepspeed"), dict):
        training_out["deepspeed"] = dict(training_block["deepspeed"])

    result["training"] = training_out

    sys_msg = global_training.get("system_msg")
    if sys_msg:
        result["system_messages"] = {"default": sys_msg}

    paths: Dict[str, Any] = {}
    if default_dataset.get("csv_train") is not None:
        paths["csv_train"] = default_dataset.get("csv_train")
    if default_dataset.get("csv_val") is not None:
        paths["csv_val"] = default_dataset.get("csv_val")
    output_dir = env_cfg.get("output_dir") if isinstance(env_cfg, dict) else None
    paths["runs_dir"] = output_dir or "runs"
    result["paths"] = paths

    if isinstance(yaml_cfg.get("deepspeed"), dict):
        result.setdefault("training", {}).setdefault("deepspeed", yaml_cfg["deepspeed"])

    return result


def load_config_dict(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a configuration file (JSON or YAML)."""
    env_path = os.environ.get("PANOVLM_CONFIG")
    cfg_path = config_path or env_path or "config.yaml"
    p = Path(cfg_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    # keep environment in sync so downstream helpers pick up the same file
    os.environ["PANOVLM_CONFIG"] = str(p)

    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Only YAML configuration files are supported. Please provide a .yaml/.yml file.")

    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configs, but it is not installed.")
    with p.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if not isinstance(raw_cfg, dict):
        raise ValueError("YAML config must deserialize to a mapping")
    cfg = _convert_yaml_config(raw_cfg)

    cfg["_config_path"] = str(p)
    logger.info(f"✓ Loaded config: {p}")
    return cfg


def get_stage_list(cfg: Dict[str, Any]):
    override = cfg.get("_cli_stage_override")
    if isinstance(override, list) and override:
        return override
    return _available_stage_names(cfg)


def resolve_stage_override(stage_arg: Optional[str], cfg: Dict[str, Any]) -> Optional[list[str]]:
    if stage_arg is None:
        return None
    stage_arg = stage_arg.strip()
    if not stage_arg or stage_arg.lower() == "all":
        return None

    tokens = [token.strip() for token in stage_arg.split(",") if token.strip()]
    if not tokens:
        return None

    available = _available_stage_names(cfg)
    lower_map = {name.lower(): name for name in available}
    resolved: list[str] = []

    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if idx < 0 or idx >= len(available):
                raise ValueError(f"Stage index {token} is out of range (1-{len(available)})")
            resolved.append(available[idx])
            continue

        key = token.lower()
        match = lower_map.get(key)
        if match is None:
            raise ValueError(f"Unknown stage '{token}'. Available stages: {available}")
        resolved.append(match)

    return resolved


def _derive_model_config_from_cfg(cfg: Dict[str, Any]) -> ModelConfig:
    pseudo_json = {
        "models": cfg.get("models", {}),
        "data": cfg.get("data", {}),
        "image_processing": cfg.get("image_processing", {}),
        "training": cfg.get("training", {}),
        "lora": cfg.get("lora", {}),
    }
    flat = ConfigManager._flatten_json_config(pseudo_json)
    return ModelConfig.from_dict(flat)

def _preview_stage_configs(cfg: Dict[str, Any]) -> None:
    planned_stages = get_stage_list(cfg)
    print("\n=== Stage Configuration Preview ===")
    print(f"Planned stages: {planned_stages}")
    for stage in planned_stages:
        merged = stage_defaults(cfg, stage)
        summary = {
            "epochs": merged.get("epochs"),
            "lr": merged.get("lr"),
            "batch_size": merged.get("batch_size"),
            "accumulate_grad_batches": merged.get("accumulate_grad_batches"),
            "data": merged.get("data"),
        }
        print(f"\n[{stage}] ->")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== Preview End ===\n")


def _ensure_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    cached = cfg.get("_model_config_obj")
    if isinstance(cached, ModelConfig):
        return cached

    model_config: Optional[ModelConfig] = None
    config_path = os.environ.get("PANOVLM_CONFIG")

    if config_path:
        try:
            model_config = ConfigManager.load_config(config_path)
        except Exception as exc:
            already_logged = cfg.get("_model_config_load_failed", False)
            if not already_logged:
                logger.debug(f"ConfigManager.load_config failed for {config_path}: {exc}")
                cfg["_model_config_load_failed"] = True

    if model_config is None:
        try:
            model_config = _derive_model_config_from_cfg(cfg)
        except Exception as exc:
            raise RuntimeError(f"Failed to derive ModelConfig from configuration: {exc}") from exc

    cfg["_model_config_obj"] = model_config
    return model_config

def stage_defaults(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """코드 기본 + 파일 설정 병합 (파일 우선)"""
    canonical = _STAGE_ALIAS_MAP.get(stage, stage)
    code_default = Config.STAGE_DEFAULTS.get(canonical, Config.STAGE_DEFAULTS.get(stage, {}))
    training_cfg = cfg.get("training", {}) or {}
    stage_configs = training_cfg.get("stage_configs")
    if isinstance(stage_configs, dict):
        file_default = stage_configs.get(stage, {}) or {}
    else:
        file_default = training_cfg.get(stage, {}) or {}
    merged = {**code_default, **file_default}

    lr_value = merged.get("lr")
    if isinstance(lr_value, str):
        try:
            merged["lr"] = float(lr_value)
        except ValueError:
            logger.warning(f"⚠️ Invalid lr value '{lr_value}' for stage '{stage}'; keeping as string")

    return merged

def _infer_default_image_size(vision_model_name: Optional[str]) -> Optional[tuple[int, int]]:
    if not vision_model_name:
        return None
    name = str(vision_model_name).lower()
    size_candidates = [
        720, 704, 640, 608, 600, 576, 560, 512, 500, 480, 448,
        432, 416, 400, 392, 384, 368, 360, 352, 336, 320,
        312, 300, 288, 272, 256, 240, 224
    ]
    for candidate in size_candidates:
        token = str(candidate)
        if token in name:
            return (candidate, candidate)
    return (224, 224)


def _resolve_stage_image_processing(cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global image_processing config with per-stage overrides."""
    base = dict(cfg.get("image_processing", {}) or {})
    # 명시적으로 제공되지 않은 경우 Vision 모델 이름을 stage-level로 허용
    models_cfg = cfg.get("models", {}) or {}
    if "vision_model_name" not in base and models_cfg.get("vision_model_name"):
        base["vision_model_name"] = models_cfg.get("vision_model_name")
    stage_overrides = None

    if isinstance(stage_cfg, dict):
        stage_overrides = stage_cfg.get("image_processing")

    if isinstance(stage_overrides, dict):
        base.update(stage_overrides)

    if not base.get("image_size") and base.get("vision_model_name"):
        inferred = _infer_default_image_size(base.get("vision_model_name"))
        if inferred:
            base["image_size"] = list(inferred)

    return base


def _normalize_data_paths(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return str(value)


def _resolve_stage_data(cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> tuple[Any, Any]:
    paths_cfg = cfg.get("paths", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    base_train = (
        _normalize_data_paths(paths_cfg.get("csv_train"))
        or _normalize_data_paths(data_cfg.get("csv_train"))
        or _normalize_data_paths(data_cfg.get("train"))
    )
    base_val = (
        _normalize_data_paths(paths_cfg.get("csv_val"))
        or _normalize_data_paths(data_cfg.get("csv_val"))
        or _normalize_data_paths(data_cfg.get("val"))
    )

    stage_train = base_train
    stage_val = base_val

    if isinstance(stage_cfg, dict):
        stage_data = stage_cfg.get("data") or {}
        if stage_data:
            train_override = (
                _normalize_data_paths(stage_data.get("csv_train"))
                or _normalize_data_paths(stage_data.get("train"))
            )
            val_override = (
                _normalize_data_paths(stage_data.get("csv_val"))
                or _normalize_data_paths(stage_data.get("val"))
            )
            if train_override is not None:
                stage_train = train_override
            if val_override is not None:
                stage_val = val_override

    return stage_train, stage_val


@dataclass(frozen=True)
class StageArtifactPaths:
    runs_dir: Path
    prefix: str
    crop: str
    resampler: str
    checkpoint_dir: Path


def _resolve_stage_artifact_paths(
    cfg: Dict[str, Any],
    stage: str,
    stage_ip: Dict[str, Any],
) -> StageArtifactPaths:
    runs_dir = Path(cfg.get("paths", {}).get("runs_dir", "runs"))
    prefix = cfg.get("training", {}).get("prefix", "panovlm")
    crop = stage_ip.get("crop_strategy", "e2p")
    resampler = (
        cfg.get("models", {}).get("resampler_type")
        or cfg.get("models", {}).get("resampler", "mlp")
    )
    ckpt_dir = runs_dir / f"{prefix}_{crop}_{stage}_{resampler}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return StageArtifactPaths(
        runs_dir=runs_dir,
        prefix=prefix,
        crop=crop,
        resampler=resampler,
        checkpoint_dir=ckpt_dir,
    )


def _to_list_str(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _save_stage_snapshot(
    cfg: Dict[str, Any],
    stage: str,
    stage_cfg: Dict[str, Any],
    image_cfg: Dict[str, Any],
    csv_train: Any,
    csv_val: Any,
) -> None:
    try:
        snapshot_dir = (
            cfg.get("paths", {}).get("stage_snapshot_dir")
            or "configs/stage_snapshots"
        )
        out_dir = Path(snapshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        training_cfg = deepcopy(stage_cfg)
        training_cfg.pop("image_processing", None)
        training_cfg.pop("data", None)

        payload = {
            "stage": stage,
            "training": training_cfg,
            "data": {
                "train": _to_list_str(csv_train),
                "val": _to_list_str(csv_val),
            },
            "image_processing": image_cfg,
            "models": cfg.get("models", {}),
            "environment": cfg.get("environment", {}),
        }

        out_path = out_dir / f"{stage}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Stage config snapshot saved: {out_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save stage snapshot for {stage}: {e}")


def build_datamodule(cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> VLMDataModule:
    ip = _resolve_stage_image_processing(cfg, stage_cfg)
    csv_train, csv_val = _resolve_stage_data(cfg, stage_cfg)

    # Vision processor가 자동 정규화를 실행하도록 mean/std가 없으면 None으로 유지
    image_mean = ip.get("image_mean", None)
    image_std = ip.get("image_std", None)
    dm = VLMDataModule(
        csv_train=csv_train,
        csv_val=csv_val,
        batch_size=stage_cfg.get("batch_size", 1),  # Tuner가 최적 크기를 찾을 것
        num_workers=cfg.get("training", {}).get("num_workers", 16),
        image_size=tuple(ip.get("image_size", [224, 224])),
        tokenizer_name=(
            cfg.get("models", {}).get("language_model_name")
            or cfg.get("models", {}).get("lm_model", "Qwen/Qwen2.5-0.5B-Instruct")
        ),
        # Allow "auto" to use tokenizer.model_max_length with cap from config
        max_text_length=stage_cfg.get("max_text_length", cfg.get("data", {}).get("max_text_length", 256)),
        crop_strategy=ip.get("crop_strategy", "e2p"),
        system_msg=cfg.get("system_messages", {}).get("default", None),
        # Image processing extras
        overlap_ratio=ip.get("overlap_ratio", 0.5),
        fov_deg=ip.get("fov_deg", 90.0),
        image_mean=image_mean,
        image_std=image_std,
        use_vision_processor=ip.get("use_vision_processor", False),
        vision_model_name=ip.get("vision_model_name", cfg.get("models", {}).get("vision_model_name")),
        anyres_patch_size=ip.get("anyres_patch_size", 336),
        anyres_max_patches=ip.get("anyres_max_patches", 12),
        normalize=ip.get("normalize", True),
        auto_max_text_length_cap=int(cfg.get("data", {}).get("auto_max_text_length_cap", 8192)),
        auto_max_text_length_floor=int(cfg.get("data", {}).get("auto_max_text_length_floor", 512)),
        auto_max_text_length_scan_limit=int(cfg.get("data", {}).get("auto_max_text_length_scan_limit", 1000)),
    )
    return dm

def build_model(cfg: Dict[str, Any], stage: str, stage_cfg: Dict[str, Any], pretrained_dir_override: Optional[str] = None) -> VLMModule:
    # ModelConfig: derive from file if available, otherwise from resolved config dict
    model_config = _ensure_model_config(cfg)

    # 학습률/LoRA만 외부로
    lr = stage_cfg.get("lr", 2e-5)
    use_lora_cfg = dict(cfg.get("lora", {}))

    stage_lora_cfg = stage_cfg.get("model_config", {}).get("lora") if isinstance(stage_cfg.get("model_config"), dict) else None
    if isinstance(stage_lora_cfg, dict) and "enabled" in stage_lora_cfg:
        use_lora_cfg["use_lora"] = bool(stage_lora_cfg.get("enabled"))
        for key in ("rank", "alpha", "dropout", "target_modules"):
            if stage_lora_cfg.get(key) is not None:
                use_lora_cfg[key] = stage_lora_cfg.get(key)

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

def build_logger_and_callbacks(
    cfg: Dict[str, Any],
    stage: str,
    stage_cfg: Dict[str, Any],
    dm: VLMDataModule,
    lit_model: VLMModule,
    stage_paths: StageArtifactPaths,
):
    ckpt_dir = stage_paths.checkpoint_dir

    # wandb
    wandb_logger = None
    try:
        # 환경변수 세팅 (보안키는 환경에서). 프로젝트명은 JSON에서 직접 읽음.
        env = cfg.get("environment", {})

        # 기존 런 닫기
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

        def _csv_name(csv_value) -> str:
            try:
                if isinstance(csv_value, (list, tuple)) and len(csv_value) > 0:
                    first = Path(str(csv_value[0]))
                    suffix = f"plus{len(csv_value)-1}" if len(csv_value) > 1 else ""
                    return f"{first.stem}{('_' + suffix) if suffix else ''}"
                return Path(str(csv_value)).stem
            except Exception:
                return "csv"

        run_name = f"{stage}_{_csv_name(dm.hparams.csv_train)}_{int(time.time())}"
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

        project_name = (
            cfg.get("training", {}).get("wandb_project")
            or cfg.get("environment", {}).get("wandb_project")
            or "panovlm"
        )
        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            config=wandb_config,
            dir="./runs",
            save_dir="./runs"
        )
    except Exception as e:
        logger.warning(f"WandB logger init failed: {e}; continue without WandB.")

    # callbacks
    callbacks = [BatchSizeMonitorCallback()]
    # EarlyStopping (메트릭 로깅 개선됨)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, mode="min", verbose=True, check_on_train_epoch_end=False
    )
    callbacks.append(early_stop)

    # ModelCheckpoint: 자동 저장 (prefix/crop/stage/resampler 기반 파일명)
    filename_base = (
        f"{stage_paths.prefix}_{stage_paths.crop}_{stage}_{stage_paths.resampler}_"
        + "{epoch:02d}-{val_loss:.4f}"
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename_base,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(ckpt_cb)

    return wandb_logger, callbacks


@dataclass
class StageResult:
    stage: str
    status: str
    best_checkpoint: Optional[str] = None
    last_checkpoint: Optional[str] = None
    artifact_dir: Optional[str] = None
    elapsed_minutes: Optional[float] = None
    error: Optional[str] = None

    def get_load_path(self) -> Optional[str]:
        for candidate in (self.best_checkpoint, self.last_checkpoint, self.artifact_dir):
            if candidate:
                return candidate
        return None


class StageExecutionError(RuntimeError):
    def __init__(self, stage: str, original_exception: Exception, result: StageResult):
        message = f"Stage '{stage}' failed: {original_exception}"
        super().__init__(message)
        self.stage = stage
        self.original_exception = original_exception
        self.result = result


def run_stage(cfg: Dict[str, Any], stage: str, prev_artifact_dir: Optional[str] = None) -> StageResult:
    logger.info(f"=== RUN STAGE: {stage} ===")

    sdef = stage_defaults(cfg, stage)
    logger.info(f"[STAGE DEFAULTS] {sdef}")

    stage_ip = _resolve_stage_image_processing(cfg, sdef)
    stage_train_data, stage_val_data = _resolve_stage_data(cfg, sdef)
    _save_stage_snapshot(cfg, stage, sdef, stage_ip, stage_train_data, stage_val_data)

    stage_paths = _resolve_stage_artifact_paths(cfg, stage, stage_ip)

    dm = build_datamodule(cfg, sdef)
    lit_model = build_model(cfg, stage, sdef, pretrained_dir_override=prev_artifact_dir)
    wandb_logger, callbacks = build_logger_and_callbacks(cfg, stage, sdef, dm, lit_model, stage_paths)

    val_check_interval_cfg = sdef.get("val_check_interval", cfg.get("training", {}).get("val_check_interval", 1.0))
    try:
        val_check_interval = float(val_check_interval_cfg)
    except (TypeError, ValueError):
        logger.warning(f"Invalid val_check_interval={val_check_interval_cfg!r}; falling back to 1.0")
        val_check_interval = 1.0

    trainer_kwargs = dict(
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        max_epochs=sdef.get("epochs", 1),
        precision="16-mixed",
        gradient_clip_val=1.0,
        default_root_dir=str(stage_paths.checkpoint_dir),
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True,
        accumulate_grad_batches=sdef.get("accumulate_grad_batches", 2),
    )

    env_cfg = cfg.get("environment", {})
    if torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        cuda_vis = str(env_cfg.get("cuda_visible_devices", "")).strip()
        if cuda_vis:
            try:
                dev_list = [int(x) for x in cuda_vis.split(",") if x.strip() != ""]
                if len(dev_list) == 1:
                    trainer_kwargs["devices"] = dev_list
                elif len(dev_list) > 1:
                    trainer_kwargs["devices"] = dev_list
            except Exception:
                pass
    else:
        trainer_kwargs["accelerator"] = "cpu"

    num_devices = 0
    if torch.cuda.is_available():
        devices_cfg = trainer_kwargs.get("devices")
        if isinstance(devices_cfg, (list, tuple)):
            num_devices = len(devices_cfg)
        elif isinstance(devices_cfg, int):
            num_devices = devices_cfg
        elif isinstance(devices_cfg, str):
            try:
                num_devices = len([d for d in devices_cfg.split(",") if d.strip()])
            except Exception:
                num_devices = torch.cuda.device_count()
        else:
            num_devices = torch.cuda.device_count()

    chosen_strategy = None
    deepspeed_cfg = (cfg.get("training", {}).get("deepspeed") or {})
    deepspeed_enabled = bool(deepspeed_cfg.get("enabled", False))
    if num_devices > 1:
        if deepspeed_enabled:
            try:
                from lightning.pytorch.strategies import DeepSpeedStrategy

                ds_kwargs = dict(deepspeed_cfg.get("strategy", {}) or {})
                if "stage" in ds_kwargs:
                    try:
                        ds_kwargs["stage"] = int(ds_kwargs["stage"])
                    except (TypeError, ValueError):
                        pass
                chosen_strategy = DeepSpeedStrategy(**ds_kwargs)
                logger.info(f"Using DeepSpeed strategy (kwargs={ds_kwargs})")
            except ImportError:
                logger.warning("DeepSpeedStrategy import failed; falling back to DDP")
            except TypeError as e:
                logger.warning(f"DeepSpeedStrategy init failed ({e}); falling back to DDP")

        if chosen_strategy is None:
            try:
                from lightning.pytorch.strategies import DDPStrategy

                chosen_strategy = DDPStrategy(find_unused_parameters=True)
                logger.info("Using DDP strategy (find_unused_parameters=True)")
            except ImportError:
                chosen_strategy = "ddp_find_unused_parameters_true"
                logger.warning("DDPStrategy import failed; using string alias 'ddp_find_unused_parameters_true'")

    if chosen_strategy is not None:
        trainer_kwargs["strategy"] = chosen_strategy
    elif deepspeed_enabled:
        logger.info("DeepSpeed enabled but only a single device detected; running without distributed strategy")

    trainer = pl.Trainer(**trainer_kwargs)

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"📊 GPU Memory after tuning - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

    logger.info(f"Starting training (stage={stage})")
    stage_exception: Optional[Exception] = None
    start_time = time.time()
    try:
        trainer.fit(lit_model, datamodule=dm)
    except Exception as exc:
        stage_exception = exc
        logger.error(f"Training failed (stage={stage}): {exc}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_minutes = round((time.time() - start_time) / 60, 3) if start_time else None
    if stage_exception is None and elapsed_minutes is not None:
        logger.info(f"Training finished in {elapsed_minutes:.1f} min")

    best_ckpt = None
    last_ckpt = None
    try:
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                if getattr(cb, "best_model_path", None):
                    best_ckpt = cb.best_model_path
                if getattr(cb, "last_model_path", None):
                    last_ckpt = cb.last_model_path
        if best_ckpt:
            logger.info(f"🏁 Best checkpoint: {best_ckpt}")
        if last_ckpt and last_ckpt != best_ckpt:
            logger.info(f"🧷 Last checkpoint: {last_ckpt}")
    except Exception as err:
        logger.warning(f"⚠️ Could not summarize checkpoints: {err}")

    canonical_stage = getattr(lit_model, "_stage_key", stage)
    if stage_exception is None and canonical_stage == "finetune" and lit_model.use_lora:
        try:
            lora_dir = str(stage_paths.checkpoint_dir / "lora_weights")
            success = lit_model.model.save_lora_weights(lora_dir)
            if success:
                logger.info(f"✓ LoRA weights saved: {lora_dir}")
            else:
                logger.warning("⚠️ LoRA weight save returned False")
        except Exception as err:
            logger.warning(f"⚠️ Additional LoRA save failed: {err}")

    if stage_exception is None:
        logger.info("=" * 80)
        logger.info("🎉 훈련 완료! 저장된 모델 사용법:")
        logger.info("=" * 80)
        if best_ckpt:
            logger.info("📖 CKPT 로딩 예시:")
            logger.info("   from panovlm.model import PanoramaVLM")
            logger.info(f'   model = PanoramaVLM.from_checkpoint("{best_ckpt}")')
        elif last_ckpt:
            logger.info("📖 CKPT 로딩 예시:")
            logger.info("   from panovlm.model import PanoramaVLM")
            logger.info(f'   model = PanoramaVLM.from_checkpoint("{last_ckpt}")')

    result = StageResult(
        stage=stage,
        status="completed" if stage_exception is None else "failed",
        best_checkpoint=best_ckpt,
        last_checkpoint=last_ckpt,
        artifact_dir=str(stage_paths.checkpoint_dir),
        elapsed_minutes=elapsed_minutes,
        error=str(stage_exception) if stage_exception else None,
    )

    if stage_exception is not None:
        raise StageExecutionError(stage, stage_exception, result) from stage_exception

    return result


class StageOrchestrator:
    STATE_VERSION = 1

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.stages = get_stage_list(cfg)
        runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
        prefix = cfg.get("training", {}).get("prefix", "panovlm")
        self.state_path = Path(runs_dir) / f"{prefix}_stage_state.json"
        self.state = self._load_state()
        self.state.setdefault("version", self.STATE_VERSION)
        self.state.setdefault("stages", {})
        force_env = os.environ.get("PANOVLM_FORCE_STAGES", "")
        self.force_stages = {s.strip() for s in force_env.split(",") if s.strip()}
        if self.force_stages:
            logger.info(f"Force rerun stages: {sorted(self.force_stages)}")
        logger.info(f"Stage state file: {self.state_path}")

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                with self.state_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                logger.warning(f"Failed to load stage state ({self.state_path}): {exc}; starting fresh.")
        return {}

    @staticmethod
    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.state_path)

    def _stage_entry(self, stage: str) -> Dict[str, Any]:
        stages = self.state.setdefault("stages", {})
        return stages.setdefault(stage, {"stage": stage})

    def _record_stage(self, stage: str, **updates) -> None:
        entry = self._stage_entry(stage)
        for key, value in updates.items():
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        entry["updated_at"] = self._now_iso()
        self._save_state()

    def _get_stage_artifact(self, stage: str) -> Optional[str]:
        entry = self.state.get("stages", {}).get(stage, {})
        for key in ("best_checkpoint", "last_checkpoint", "artifact_dir"):
            path = entry.get(key)
            if path:
                return path
        return None

    def _resolve_resume_checkpoint(self, stage: str, upstream_checkpoint: Optional[str]) -> Optional[str]:
        entry = self.state.get("stages", {}).get(stage, {})
        for key in ("resume_checkpoint", "best_checkpoint", "last_checkpoint", "artifact_dir"):
            candidate = entry.get(key)
            if candidate:
                return candidate
        return upstream_checkpoint

    def _should_skip_stage(self, stage: str) -> bool:
        entry = self.state.get("stages", {}).get(stage, {})
        return entry.get("status") == "completed" and stage not in self.force_stages

    def run(self) -> Optional[str]:
        logger.info(f"Planned stages: {self.stages}")
        prev_artifact = None
        for stage in self.stages:
            if self._should_skip_stage(stage):
                artifact = self._get_stage_artifact(stage)
                prev_artifact = artifact or prev_artifact
                logger.info(f"Skipping stage '{stage}' (already completed). Using checkpoint: {artifact}")
                continue

            resume_candidate = self._resolve_resume_checkpoint(stage, prev_artifact)
            self._record_stage(
                stage,
                status="running",
                started_at=self._now_iso(),
                error=None,
                upstream_checkpoint=prev_artifact,
                resume_checkpoint=resume_candidate,
            )

            try:
                result = run_stage(self.cfg, stage, prev_artifact_dir=resume_candidate)
            except StageExecutionError as err:
                result = err.result
                self._record_stage(
                    stage,
                    status="failed",
                    error=str(err.original_exception),
                    failed_at=self._now_iso(),
                    best_checkpoint=result.best_checkpoint,
                    last_checkpoint=result.last_checkpoint,
                    artifact_dir=result.artifact_dir,
                    elapsed_minutes=result.elapsed_minutes,
                )
                raise err
            else:
                self._record_stage(
                    stage,
                    status=result.status,
                    completed_at=self._now_iso(),
                    best_checkpoint=result.best_checkpoint,
                    last_checkpoint=result.last_checkpoint,
                    artifact_dir=result.artifact_dir,
                    elapsed_minutes=result.elapsed_minutes,
                    error=None,
                    resume_checkpoint=None,
                )
                prev_artifact = result.get_load_path() or prev_artifact

                # 각 단계 완료 후 시각화 실행
                _run_stage_visualization(self.cfg, stage, result.best_checkpoint)

        return prev_artifact


def run_all(cfg: Dict[str, Any]) -> Optional[str]:
    orchestrator = StageOrchestrator(cfg)
    final_artifact = orchestrator.run()
    if final_artifact:
        logger.info(f"Pipeline finished. Final artifact: {final_artifact}")
    else:
        logger.info("Pipeline finished without checkpoint artifacts.")
    return final_artifact

# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def _parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panorama VLM training")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (JSON or YAML).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default='vision_pretraining, resampler_training, instruction_tuning',
        help="Comma-separated stage names or 1-based indices to run (e.g. 'vision,resampler' or '1,2').",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print resolved stage configurations and exit.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_cli_arguments()
    cfg = load_config_dict(args.config)
    try:
        stage_override = resolve_stage_override(args.stage, cfg)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)
    if stage_override:
        cfg["_cli_stage_override"] = stage_override
        logger.info(f"CLI stage override → {stage_override}")

    if getattr(args, "preview", False):
        _preview_stage_configs(cfg)
        sys.exit(0)
    try:
        run_all(cfg)
    except StageExecutionError as err:
        logger.error(f"Stage orchestration failed: {err}")
        sys.exit(1)
