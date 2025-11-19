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

# HuggingFace 캐시 최적화 설정 (메모리 절약)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys
import json
import time
import gc
import shutil
import subprocess
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
from panovlm.dataset   import VLMDataModule
from panovlm.models.model import PanoramaVLM
from panovlm.utils     import *
from panovlm.config    import ModelConfig, PanoVLMConfig
from panovlm.runtime import (
    StageManager,
    canonical_stage_name,
    ModelFactory,
    load_runtime_config,
)
# ----------------------------------------------------------------------------

# ── 로깅 설정 ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")]
)
logger = logging.getLogger("panovlm.train")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVAL_SCRIPT_PATH = SCRIPT_DIR / "eval.py"


# ─────────────────────────────────────────────────────────────────────────────
# Experiment naming helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short_id_from_model_path(model_name: Optional[str], max_len: int = 12, *, siglip_include_patch_res: bool = False) -> str:
    """Extract a compact identifier from a HF-style model name.

    Examples:
      - "google/siglip2-so400m-patch16-256" -> "siglip2"
      - "Qwen/Qwen3-0.6B" -> "Qwen3"
    """
    if not model_name:
        return "model"
    base = str(model_name).split("/")[-1]
    base_lower = base.lower()

    # Special-case SigLIP family: keep first two tokens; optionally add _pXX_YYY
    if base_lower.startswith("siglip"):
        parts = base.split("-")
        first = parts[0] if parts else "siglip"
        second = parts[1] if len(parts) > 1 else None
        core = first if not second else f"{first}-{second}"

        if siglip_include_patch_res and len(parts) > 2:
            import re as _re
            patch_num = None
            res_num = None
            for p in parts[2:]:
                pl = p.lower()
                m = _re.match(r"patch(\d+)", pl)
                if m:
                    patch_num = m.group(1)
                elif pl.isdigit():
                    res_num = pl
            if patch_num and res_num:
                core = f"{core}_p{patch_num}_{res_num}"
        return core

    # Generic fallback: first token before '-' or '_' and keep alnum prefix
    token = base.split("-")[0].split("_")[0]
    import re as _re
    m = _re.match(r"([A-Za-z]+\d+)", token)
    if m:
        token = m.group(1)
    token = token[:max_len]
    return token


def _compute_experiment_name(cfg: Dict[str, Any], crop_strategy: Optional[str] = None) -> str:
    """Build a readable experiment name from config components unless explicitly provided.

    Pattern: {VISION}_{LM}_{RESAMPLER}_{CROP}{_PE}
    - VISION: short id from models.vision_name
    - LM:     short id from models.language_model_name
    - RESAMPLER: models.resampler_type (fallback models.resampler)
    - CROP:   image_processing.crop_strategy, '_' -> '-'
    - PE:     append '_PE' if models.use_projection_positional_encoding is True
    """
    exp_cfg = cfg.get("experiment", {}) or {}

    name_from_cfg = exp_cfg.get("name")
    auto_flag = bool(exp_cfg.get("auto_name", False))
    if isinstance(name_from_cfg, str) and name_from_cfg.strip() and name_from_cfg.strip().lower() not in {"auto", "{auto}"} and not auto_flag:
        return name_from_cfg.strip()
    # Legacy fallback: training.prefix (unless auto_name is explicitly requested)
    if (not name_from_cfg or not str(name_from_cfg).strip() or str(name_from_cfg).strip().lower() in {"auto", "{auto}"}) and not auto_flag:
        legacy_prefix = (cfg.get("training", {}) or {}).get("prefix")
        if isinstance(legacy_prefix, str) and legacy_prefix.strip():
            return legacy_prefix.strip()

    # Compose from components
    models_cfg = cfg.get("models", {}) or {}
    vision_full = models_cfg.get("vision_name")
    lm_full = models_cfg.get("language_model_name")
    resampler = models_cfg.get("resampler_type") or models_cfg.get("resampler", "mlp")
    crop = crop_strategy or (cfg.get("image_processing", {}) or {}).get("crop_strategy", "e2p")
    crop_short = str(crop).replace("_", "-")
    use_pe = bool(models_cfg.get("use_projection_positional_encoding", False))

    siglip_inc = bool((cfg.get("experiment", {}) or {}).get("siglip_include_patch_res", False))
    vision_short = _short_id_from_model_path(vision_full, max_len=12, siglip_include_patch_res=siglip_inc)
    lm_short = _short_id_from_model_path(lm_full, max_len=12)

    parts = [vision_short, lm_short, resampler, crop_short]
    if use_pe:
        parts.append("PE")
    exp_name = "_".join(parts)

    # Sanitize to filesystem-friendly name
    exp_name = exp_name.replace(" ", "_")
    # Keep only [A-Za-z0-9_\-]
    import re as _re
    exp_name = _re.sub(r"[^A-Za-z0-9_\-]", "", exp_name)
    return exp_name

# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class VLMModule(pl.LightningModule):
    """Panorama VLM Lightning 래퍼 (stage-aware)"""

    def __init__(self, *, stage: str, model_config: ModelConfig, lr: float,
                 use_lora_cfg: Dict[str, Any], pretrained_dir: Optional[str] = None,
                 vision_trainable_blocks: int = 0, cache_cleanup_interval: int = 1000):
        super().__init__()
        
        # ModelConfig는 별도 저장 (직렬화 불가능할 수 있음)
        self.model_config: ModelConfig = model_config
        
        # Lightning 권장 방식: 초기화 시점에 모든 hparams 저장
        # 체크포인트 메타데이터를 포함하여 한 번에 저장
        # Note: max_text_length는 VLMDataModule에서 관리하므로 여기서는 제외 (Lightning hparams 충돌 방지)
        checkpoint_metadata = {
            # 훈련 설정
            "stage": stage,
            "lr": lr,
            "vision_trainable_blocks": vision_trainable_blocks,
            "cache_cleanup_interval": cache_cleanup_interval,
            "use_lora": bool(use_lora_cfg.get("use_lora", False)),
            "lora_rank": use_lora_cfg.get("rank", 16),
            "lora_alpha": use_lora_cfg.get("alpha", 32),
            "lora_dropout": use_lora_cfg.get("dropout", 0.1),
            "pretrained_dir": pretrained_dir,
            # 모델 설정 (복원에 필요)
            "vision_name": model_config.vision_name,
            "language_model_name": model_config.language_model_name,
            "resampler_type": model_config.resampler_type,
            "latent_dimension": model_config.latent_dimension,
            "vicreg_loss_weight": model_config.vicreg_loss_weight,
            # Save model-specific overlap under distinct key to avoid Lightning merging
            # conflicts when DataModule also exposes an 'overlap_ratio' that may be
            # intentionally different. PanoramaVLM.from_checkpoint will accept both.
            "model_overlap_ratio": model_config.overlap_ratio,
            # max_text_length는 VLMDataModule의 hparams에 이미 저장됨 (충돌 방지를 위해 제외)
        }
        self.save_hyperparameters(checkpoint_metadata)
        
        # 편의성을 위한 속성들
        self.lr = lr
        self.learning_rate = lr  # Lightning Tuner를 위한 속성
        self.vision_trainable_blocks = vision_trainable_blocks
        self.cache_cleanup_interval = cache_cleanup_interval

        # 모델 생성 우선순위: pretrained_dir(.ckpt 또는 HF 디렉토리) > scratch
        self.model_factory = ModelFactory(self.model_config)

        if pretrained_dir and os.path.isdir(pretrained_dir):
            logger.info(f"🧩 Loading from pretrained dir: {pretrained_dir}")
            try:
                self.model = self.model_factory.load_pretrained_dir(pretrained_dir)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load pretrained dir ({pretrained_dir}): {e}. Falling back to scratch init.")
                self.model = self.model_factory.build()
        elif pretrained_dir and os.path.isfile(pretrained_dir) and str(pretrained_dir).endswith('.ckpt'):
            logger.info(f"🧩 Loading from checkpoint file: {pretrained_dir}")
            try:
                self.model = self.model_factory.load_checkpoint(pretrained_dir)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load checkpoint file ({pretrained_dir}): {e}. Falling back to scratch init.")
                self.model = self.model_factory.build()
        else:
            self.model = self.model_factory.build()

        # stage 검증/매핑
        try:
            self._stage_key = canonical_stage_name(stage)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if stage != self._stage_key:
            logger.info(f"Stage alias resolved: '{stage}' → '{self._stage_key}'")

        # LoRA 설정 (finetune에서만 적용)
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
        self._unfreeze_for_stage(self._stage_key, vision_trainable_blocks=self.vision_trainable_blocks)
        
        logger.info(f"✓ VLMModule 초기화 완료 (stage={self._stage_key}, LoRA={self.use_lora})")

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
                return torch.tensor(0.0, device=self.device, requires_grad=True)

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

            # 주기적 캐시 정리 (메모리 누수 방지) - 설정 가능
            if self.cache_cleanup_interval > 0 and batch_idx % self.cache_cleanup_interval == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

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
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                logger.error(f"Runtime error in training step {self.global_step}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import traceback
                logger.error("Traceback:\n" + traceback.format_exc())
                # 스텝 대칭 유지
                return torch.tensor(0.0, device=self.device, requires_grad=True)
        except Exception as e:
            logger.error(f"Unexpected error in training step {self.global_step}: {e}")
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
            # 스텝 대칭 유지
            return torch.tensor(0.0, device=self.device, requires_grad=True)

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
                return torch.tensor(0.0, device=self.device, requires_grad=True)  # 스칼라 텐서 반환

            loss = out["loss"]

            if not torch.isfinite(loss):
                logger.warning(f"[VAL] Non-finite val loss at step {batch_idx}: {loss}")
                return torch.tensor(0.0, device=self.device, requires_grad=True)  # 스칼라 텐서 반환

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
                return torch.tensor(0.0, device=self.device, requires_grad=True)  # 스칼라 텐서 반환
            else:
                logger.error(f"[VAL] Runtime error in validation step {batch_idx}: {e}")
                import traceback
                logger.error("Traceback:\n" + traceback.format_exc())
                return torch.tensor(0.0, device=self.device, requires_grad=True)  # 스칼라 텐서 반환
        except Exception as e:
            logger.error(f"[VAL] Error in validation step {batch_idx}: {e}")
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # 스칼라 텐서 반환

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
    def _set_vision_trainable_blocks(self, num_blocks: int = 0):
        """Vision encoder의 마지막 N개 블록만 학습 가능하도록 설정

        Args:
            num_blocks: 학습할 마지막 블록 수
                       0 = 전체 freeze (기본값)
                       -1 = 전체 unfreeze
                       N > 0 = 마지막 N개 블록만 unfreeze
        """
        vision = getattr(self.model, "vision_backbone", None)
        if vision is None:
            logger.warning("⚠️ vision_backbone not found")
            return

        # VisionBackbone wraps the actual encoder in .encoder attribute
        # Get the actual vision encoder (SigLIP, CLIP, etc.)
        encoder = getattr(vision, "encoder", vision)

        layers = None

        # Try encoder.encoder.layers (SigLIP: SiglipVisionTransformer.encoder.layers)
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
            layers = list(encoder.encoder.layers)
        # Try encoder.vision_model.encoder.layers (CLIP structure)
        elif hasattr(encoder, "vision_model"):
            vision_model = encoder.vision_model
            if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
                layers = list(vision_model.encoder.layers)
        # Try encoder.layers (some other architectures)
        elif hasattr(encoder, "layers"):
            layers = list(encoder.layers)

        if layers is None or len(layers) == 0:
            logger.warning(f"⚠️ Could not find layers in vision encoder structure: {type(vision)}")
            logger.debug(f"Available attributes: {dir(vision)}")
            return

        total_layers = len(layers)

        if num_blocks == -1:
            # Unfreeze all layers
            for layer in layers:
                layer.requires_grad_(True)
            logger.info(f"✓ All {total_layers} vision encoder layers unfrozen")
        elif num_blocks > 0:
            # Unfreeze last N blocks
            num_blocks = min(num_blocks, total_layers)
            for layer in layers[-num_blocks:]:
                layer.requires_grad_(True)
            logger.info(f"✓ Last {num_blocks}/{total_layers} vision encoder layers unfrozen")
        else:
            logger.info(f"✓ All {total_layers} vision encoder layers remain frozen")
            
    def _unfreeze_for_stage(self, stage: str, vision_trainable_blocks: int = 0):
        """각 stage에 맞게 파라미터를 freeze/unfreeze

        Args:
            stage: 학습 단계 ("vision", "resampler", "finetune")
            vision_trainable_blocks: Vision encoder에서 학습할 블록 수
                                    0 = 전체 freeze (기본값)
                                    -1 = 전체 unfreeze
                                    N > 0 = 마지막 N개 블록 unfreeze
        """
        # 모든 파라미터를 freeze한 뒤, 필요한 부분만 unfreeze
        self.model.requires_grad_(False)

        if stage == "vision":
            # VICReg stage: Vision Encoder (optional partial) → Resampler (trainable) → VICReg Projector (trainable)
            # Resampler learns to produce meaningful representations via contrastive learning

            # Optionally unfreeze vision encoder layers
            if vision_trainable_blocks != 0:
                self._set_vision_trainable_blocks(vision_trainable_blocks)

            # Always unfreeze resampler and vicreg_projector
            if hasattr(self.model, "resampler"):
                self.model.resampler.requires_grad_(True)
                logger.info("✓ Resampler unfrozen for VICReg training")
            if hasattr(self.model, "vicreg_projector"):
                self.model.vicreg_projector.requires_grad_(True)
                logger.info("✓ VICReg projector unfrozen")
            else:
                logger.warning("⚠️ vicreg_projector not found - vision stage may not train properly")

            if vision_trainable_blocks == 0:
                logger.info("✓ Stage 1: Resampler + VICReg projector trainable (vision encoder fully frozen)")
            else:
                logger.info(f"✓ Stage 1: Vision encoder (partial) + Resampler + VICReg projector trainable")
        elif stage == "resampler":
            # Resampler stage: Optionally unfreeze more vision layers, always unfreeze resampler + projection

            # Optionally unfreeze vision encoder layers
            if vision_trainable_blocks != 0:
                self._set_vision_trainable_blocks(vision_trainable_blocks)

            # Always unfreeze resampler and projection
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)

            if vision_trainable_blocks == 0:
                logger.info("✓ Stage 2: Resampler + Projection unfrozen (vision encoder frozen)")
            else:
                logger.info(f"✓ Stage 2: Vision encoder (partial) + Resampler + Projection unfrozen")
        elif stage == "finetune":
            # Finetune stage: Optionally unfreeze vision layers, always unfreeze resampler + projection

            # Optionally unfreeze vision encoder layers
            if vision_trainable_blocks != 0:
                self._set_vision_trainable_blocks(vision_trainable_blocks)

            # Always unfreeze resampler and projection
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)

            # LM은 항상 freeze (LoRA 사용 여부와 무관)
            for p in self.model.language_model.parameters():
                p.requires_grad = False

            if vision_trainable_blocks == 0:
                logger.info("✓ Stage 3: Resampler + Projection unfrozen (vision encoder frozen, LM frozen/LoRA)")
            else:
                logger.info(f"✓ Stage 3: Vision encoder (partial) + Resampler + Projection unfrozen (LM frozen/LoRA)")

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


# ─────────────────────────────────────────────────────────────────────────────
# 콜백: 간단 모니터링 및 메타데이터 관리
# ─────────────────────────────────────────────────────────────────────────────
class MetadataCallback(pl.Callback):
    """체크포인트와 함께 메타데이터 및 구성 파일을 저장하고 심볼릭 링크를 관리"""

    def __init__(self, ckpt_dir: str, metadata: Dict[str, Any], config_path: Optional[str] = None):
        self.ckpt_dir = Path(ckpt_dir)
        self.metadata = metadata
        self.meta_path = self.ckpt_dir / "checkpoint_metadata.json"
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self._config_copied = False

        if self.config_path:
            self.metadata.setdefault("config", {})
            self.metadata["config"]["source_path"] = str(self.config_path)
            self.metadata["config"]["saved_filename"] = "config.yaml"

        self._ensure_config_copy()

    def _ensure_config_copy(self) -> None:
        """Copy the YAML config into the checkpoint directory for downstream use."""
        if not self.config_path or self._config_copied:
            return

        try:
            if not self.config_path.exists():
                logger.warning(f"⚠️ Config file does not exist, skip copy: {self.config_path}")
                return

            target_path = self.ckpt_dir / "config.yaml"
            if target_path.exists():
                # Skip copying if the target already points to the same file contents
                try:
                    if target_path.resolve() == self.config_path:
                        self._config_copied = True
                        return
                except Exception:
                    pass

            shutil.copy2(self.config_path, target_path)
            self._config_copied = True
            logger.debug(f"✓ Config copied to checkpoint dir: {target_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to copy config file to checkpoint dir: {e}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """체크포인트 저장 시 메타데이터도 함께 저장"""
        if not self._config_copied:
            self._ensure_config_copy()

        try:
            # 현재 학습 상태 정보 추가
            full_meta = {
                **self.metadata,
                "epoch_info": {
                    "current_epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
                },
                "checkpoint_info": {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "pytorch_lightning_version": pl.__version__,
                    "torch_version": torch.__version__,
                }
            }
            
            # JSON 저장
            with self.meta_path.open("w", encoding="utf-8") as f:
                json.dump(full_meta, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✓ Metadata saved: {self.meta_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save metadata: {e}")
    
    


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
        logger.info(f"[DATA] crop={trainer.datamodule.hparams.crop_strategy}")
        logger.info(f"[DATA] vision_model_name={trainer.datamodule.hparams.vision_model_name}")
        logger.info(f"[DATA] use_vision_processor={trainer.datamodule.hparams.use_vision_processor}")
        logger.info(f"[DATA] Actual processor image_size={trainer.datamodule.processor.img_proc.image_size}")
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

def load_config_dict(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Backward-compatible shim for modules that import this helper from train.py."""
    bundle = load_runtime_config(config_path)
    cfg = bundle.raw
    cfg["_pano_config_obj"] = bundle.pano
    cfg["_model_config_obj"] = bundle.model
    if "_config_path" in cfg:
        logger.info(f"✓ Loaded config: {cfg['_config_path']}")
    return cfg


def _validate_required_model_fields(cfg: Dict[str, Any]) -> None:
    models_cfg = cfg.get("models")
    if not isinstance(models_cfg, dict):
        raise ValueError("YAML config에 'models' 섹션이 필요합니다. vision/language/resampler를 명시하세요.")
    required = ("vision_name", "language_model_name", "resampler_type")
    missing = [key for key in required if not models_cfg.get(key)]
    if missing:
        raise ValueError(f"models 섹션에 필수 파라미터가 없습니다: {missing}. YAML을 업데이트하세요.")

    # Stage configuration structure sanity check
    training_cfg = cfg.get("training", {}) or {}
    stage_cfgs = training_cfg.get("stage_configs", {})
    if not isinstance(stage_cfgs, dict) or not stage_cfgs:
        raise ValueError("training.stage_configs가 비어 있습니다. 스테이지별 설정을 YAML로 정의하세요.")
    for stage_name, stage_def in stage_cfgs.items():
        if not isinstance(stage_def, dict):
            raise ValueError(f"stage '{stage_name}' 설정이 올바르지 않습니다. dict 형태로 정의하세요.")


def _preview_stage_configs(stage_manager: StageManager) -> None:
    planned_stages = stage_manager.available_stage_names()
    print("\n=== Stage Configuration Preview ===")
    print(f"Planned stages: {planned_stages}")
    for summary in stage_manager.preview():
        stage = summary.pop("stage")
        print(f"\n[{stage}] ->")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== Preview End ===\n")


def _ensure_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    _validate_required_model_fields(cfg)
    cached = cfg.get("_model_config_obj")
    if isinstance(cached, ModelConfig):
        return cached

    pano_cfg = cfg.get("_pano_config_obj")
    if isinstance(pano_cfg, PanoVLMConfig):
        model_config = pano_cfg.models
    else:
        try:
            pano_cfg = PanoVLMConfig(**cfg)
        except Exception as exc:
            raise RuntimeError("Failed to construct PanoVLMConfig from configuration") from exc
        cfg["_pano_config_obj"] = pano_cfg
        model_config = pano_cfg.models

    cfg["_model_config_obj"] = model_config
    return model_config

def _resolve_stage_image_processing(cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global image_processing config with per-stage overrides."""
    base = dict(cfg.get("image_processing", {}) or {})
    # 명시적으로 제공되지 않은 경우 Vision 모델 이름을 stage-level로 허용
    # 주의: YAML과 ModelConfig에서는 'vision_name', train.py 일부에서는 'vision_model_name' 사용
    # 두 가지 모두 지원하되, 'vision_name'을 우선
    models_cfg = cfg.get("models", {}) or {}
    if "vision_model_name" not in base:
        # 우선순위: vision_name > vision_model_name
        vision_identifier = models_cfg.get("vision_name") or models_cfg.get("vision_model_name")
        if vision_identifier:
            base["vision_model_name"] = vision_identifier
    stage_overrides = None

    if isinstance(stage_cfg, dict):
        stage_overrides = stage_cfg.get("image_processing")

    if isinstance(stage_overrides, dict):
        base.update(stage_overrides)

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
    
    # image_size 처리: None이면 PanoramaImageProcessor가 자동 추론
    image_size_value = ip.get("image_size")
    if image_size_value is not None:
        image_size_tuple = tuple(image_size_value)
    else:
        image_size_tuple = None  # PanoramaImageProcessor가 vision_model_name에서 추론
    
    dm = VLMDataModule(
        csv_train=csv_train,
        csv_val=csv_val,
        batch_size=stage_cfg.get("batch_size", 1),  # Tuner가 최적 크기를 찾을 것
        num_workers=cfg.get("training", {}).get("num_workers", 16),
        image_size=image_size_tuple,  # None 허용
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
        anyres_patch_size=ip.get("anyres_patch_size"),  # None이면 image_size에서 자동 추론
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

    # Vision encoder trainable blocks 설정 (stage config에서 읽기)
    vision_trainable_blocks = stage_cfg.get("vision_trainable_blocks", 0)

    # 캐시 정리 간격 설정 (training config에서 읽기)
    cache_cleanup_interval = cfg.get("training", {}).get("cache_cleanup_interval", 1000)

    module = VLMModule(
        stage=stage,
        model_config=model_config,
        lr=lr,
        use_lora_cfg=use_lora_cfg,
        pretrained_dir=pretrained_dir,
        vision_trainable_blocks=vision_trainable_blocks,
        cache_cleanup_interval=cache_cleanup_interval,
    )
    return module

def build_logger_and_callbacks(cfg: Dict[str, Any], stage: str, stage_cfg: Dict[str, Any], dm: VLMDataModule, lit_model: VLMModule):
    """WandB logger와 콜백 생성 (체크포인트 저장 경로 및 파일명 포함)
    
    이름 생성 규칙 (단일 장소에서 관리):
    =====================================================
    1. experiment_name: YAML의 experiment.name 필드
       예: "ADDDATA_S2Q3_1_latent768_PE"
       
    2. 디렉토리 구조:
       runs/{experiment_name}/{stage}/{crop_short}_{resampler}/
       예: runs/ADDDATA_S2Q3/vision/anyres-e2p_mlp/
       
    3. 체크포인트 파일명:
       {vision_short}_{resampler}_{crop_short}_{dataset}_epoch{XX}_loss{Y.YYYY}.ckpt
       예: siglip_mlp_anyres-e2p_quic360_epoch03_loss0.4523.ckpt
       
    4. WandB Run Name:
       {experiment_name}/{stage}/{vision_short}_{resampler}_{crop_short}_{dataset}_{timestamp}
       예: ADDDATA_S2Q3/vision/siglip_mlp_anyres-e2p_quic360_1015-1430
    =====================================================
    """
    
    # ========== 공통 이름 구성 요소 생성 (단일 장소에서 관리) ==========
    def _csv_name(csv_value) -> str:
        """CSV 경로에서 데이터셋 이름 추출"""
        try:
            if isinstance(csv_value, (list, tuple)) and len(csv_value) > 0:
                first = Path(str(csv_value[0]))
                suffix = f"plus{len(csv_value)-1}" if len(csv_value) > 1 else ""
                return f"{first.stem}{('_' + suffix) if suffix else ''}"
            return Path(str(csv_value)).stem
        except Exception:
            return "csv"
    
    # YAML에서 가져온 experiment 이름 (없거나 auto면 구성요소로 생성)
    # YAML 예시: experiment: { name: "ADDDATA_S2Q3_1_latent768_PE" }
    experiment_name = _compute_experiment_name(cfg, crop_strategy=dm.hparams.crop_strategy)
    
    # 모델 구성 요소
    vision_full = cfg.get("models", {}).get("vision_name", "unknown")
    vision_short = vision_full.split("/")[-1].split("-")[0][:10]  # "google/siglip-base-patch16-224" -> "siglip"
    
    resampler = (
        cfg.get("models", {}).get("resampler_type")
        or cfg.get("models", {}).get("resampler", "mlp")
    )
    
    # 이미지 처리 전략
    crop_strategy = dm.hparams.crop_strategy
    crop_short = crop_strategy.replace("_", "-")  # anyres_e2p -> anyres-e2p
    
    # 데이터셋 이름
    dataset_name = _csv_name(dm.hparams.csv_train)
    
    # ========== 디렉토리 및 파일 경로 ==========
    runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
    
    # 체크포인트 디렉토리: {experiment_name}/{stage}/{crop_strategy}_{resampler}
    # 예: runs/ADDDATA_S2Q3/vision/anyres-e2p_mlp/
    ckpt_dir = f"{runs_dir}/{experiment_name}/{stage}/{crop_short}_{resampler}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

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

        # ========== WandB Run Name ==========
        # 날짜/시간 추가 (간결하게: MMDD-HHMM)
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d-%H%M")
        
        # 형식: {experiment_name}/{stage}/{vision}_{resampler}_{crop}_{dataset}_{timestamp}
        # 예: ADDDATA_SQ3/vision/siglip_mlp_anyres-e2p_quic360_1015-1430
        run_name = f"{experiment_name}/{stage}/{vision_short}_{resampler}_{crop_short}_{dataset_name}_{timestamp}"
        
        # WandB Config: 하이퍼파라미터 및 실험 설정 상세 기록
        wandb_config = {
            # Stage & Experiment
            "experiment_name": experiment_name,
            "stage": stage,
            "stage_canonical": canonical_stage_name(stage),
            
            # Model Architecture
            "vision_encoder": vision_full,
            "language_model": cfg.get("models", {}).get("language_model_name"),
            "resampler_type": resampler,
            "latent_dimension": lit_model.model_config.latent_dimension,
            "vision_trainable_blocks": lit_model.vision_trainable_blocks,
            
            # Training Hyperparameters
            "learning_rate": lit_model.hparams.lr,
            "batch_size": dm.hparams.batch_size,
            "accumulate_grad_batches": stage_cfg.get("accumulate_grad_batches", 1),
            "epochs": stage_cfg.get("epochs"),
            "gradient_clip_val": 1.0,
            "optimizer": "AdamW",
            "weight_decay": 0.05,
            
            # Image Processing
            "image_size": dm.hparams.image_size,
            "crop_strategy": dm.hparams.crop_strategy,
            "fov_deg": dm.hparams.fov_deg,
            "overlap_ratio": dm.hparams.overlap_ratio,
            "use_vision_processor": dm.hparams.use_vision_processor,
            "normalize": dm.hparams.normalize,
            
            # Text Processing
            "max_text_length": dm.hparams.max_text_length,
            "tokenizer": dm.hparams.tokenizer_name,
            
            # Dataset
            "train_dataset": dataset_name,
            "num_train_samples": len(dm.train_ds) if hasattr(dm, 'train_ds') and dm.train_ds else 0,
            "num_val_samples": len(dm.val_ds) if hasattr(dm, 'val_ds') and dm.val_ds else 0,
            "num_workers": dm.hparams.num_workers,
            
            # LoRA (if applicable)
            "use_lora": lit_model.use_lora,
            "lora_rank": lit_model.hparams.lora_rank if lit_model.use_lora else None,
            "lora_alpha": lit_model.hparams.lora_alpha if lit_model.use_lora else None,
            
            # VICReg (for vision stage)
            "vicreg_loss_weight": lit_model.model_config.vicreg_loss_weight if stage == "vision" else None,
            
            # System
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mixed_precision": "bf16-mixed",
        }

        # WandB Tags: 빠른 필터링을 위한 태그
        wandb_tags = [
            stage,  # 스테이지별 필터링
            resampler,  # Resampler 타입별
            crop_short,  # Crop 전략별
            dataset_name,  # 데이터셋별
            vision_short,  # Vision 모델별
        ]
        if lit_model.use_lora:
            wandb_tags.append("lora")
        if stage == "vision":
            wandb_tags.append("vicreg")
        
        # WandB Notes: 실험 설명
        wandb_notes = f"""
        Stage: {stage}
        Vision: {cfg.get("models", {}).get("vision_name")}
        LM: {cfg.get("models", {}).get("language_model_name")}
        Resampler: {resampler}
        Dataset: {dataset_name}
        Image Size: {dm.hparams.image_size}
        Crop Strategy: {dm.hparams.crop_strategy}
        """

        project_name = (
            cfg.get("training", {}).get("wandb_project")
            or cfg.get("environment", {}).get("wandb_project")
            or "panovlm"
        )
        
        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=wandb_tags,
            notes=wandb_notes.strip(),
            dir="./runs",
            save_dir="./runs",
            log_model=False,  # 체크포인트는 ModelCheckpoint로 관리
        )
    except Exception as e:
        logger.warning(f"WandB logger init failed: {e}; continue without WandB.")

    # ========== 체크포인트 메타데이터 준비 ==========
    checkpoint_metadata: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "stage": stage,
        "stage_canonical": canonical_stage_name(stage),
        
        "model_config": {
            "vision_name": vision_full,
            "language_model_name": cfg.get("models", {}).get("language_model_name"),
            "resampler_type": resampler,
            "latent_dimension": lit_model.model_config.latent_dimension,
            "image_size": list(dm.hparams.image_size) if isinstance(dm.hparams.image_size, tuple) else dm.hparams.image_size,
            "max_text_length": dm.hparams.max_text_length,
            "vicreg_loss_weight": lit_model.model_config.vicreg_loss_weight,
            "overlap_ratio": lit_model.model_config.overlap_ratio,
            # ✨ Resampler 상세 설정 추가 (dimension mismatch 방지용)
            "resampler_config": getattr(lit_model.model_config, 'resampler_config', None),
            "resampler_hidden_dim": getattr(lit_model.model_config, 'resampler_hidden_dim', None),
        },
        
        "training_config": {
            "learning_rate": lit_model.hparams.lr,
            "batch_size": dm.hparams.batch_size,
            "accumulate_grad_batches": stage_cfg.get("accumulate_grad_batches", 1),
            "epochs": stage_cfg.get("epochs"),
            "crop_strategy": crop_strategy,
            "fov_deg": dm.hparams.fov_deg,
            "overlap_ratio": dm.hparams.overlap_ratio,
            "use_vision_processor": dm.hparams.use_vision_processor,
            "normalize": dm.hparams.normalize,
            "use_lora": lit_model.use_lora,
            "lora_rank": lit_model.hparams.lora_rank if lit_model.use_lora else None,
            "lora_alpha": lit_model.hparams.lora_alpha if lit_model.use_lora else None,
            "vision_trainable_blocks": lit_model.vision_trainable_blocks,
        },
        
        "dataset": {
            "train_csv": str(dm.hparams.csv_train),
            "val_csv": str(dm.hparams.csv_val),
            "dataset_name": dataset_name,
            "num_train_samples": len(dm.train_ds) if hasattr(dm, 'train_ds') and dm.train_ds else 0,
            "num_val_samples": len(dm.val_ds) if hasattr(dm, 'val_ds') and dm.val_ds else 0,
            "num_workers": dm.hparams.num_workers,
        },
        
        "system": {
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mixed_precision": "bf16-mixed",
        },
        
        "wandb": {
            "project": wandb_logger.experiment.project if wandb_logger else None,
            "run_name": run_name if wandb_logger else None,
            "run_id": wandb_logger.experiment.id if wandb_logger and hasattr(wandb_logger.experiment, 'id') else None,
        } if wandb_logger else None,
    }
    if cfg.get("_config_path"):
        checkpoint_metadata.setdefault("config", {})
        checkpoint_metadata["config"].update(
            {
                "source_path": cfg["_config_path"],
                "saved_filename": "config.yaml",
            }
        )

    # callbacks
    callbacks = [
        BatchSizeMonitorCallback(),
        MetadataCallback(ckpt_dir, checkpoint_metadata, config_path=cfg.get("_config_path")),  # 메타데이터 및 config 저장
    ]
    
    # EarlyStopping (메트릭 로깅 개선됨)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, mode="min", verbose=True, check_on_train_epoch_end=False
    )
    callbacks.append(early_stop)

    # ========== ModelCheckpoint: 가독성 높은 파일명으로 저장 ==========
    # save_weights_only=True: 모델 가중치만 저장 (optimizer/scheduler 상태 제외)
    # → 체크포인트 크기 감소, 로딩 속도 개선
    # → 훈련 재개가 아닌 inference/다음 stage 용도로 충분
    
    # 파일명 형식: {vision}_{resampler}_{crop}_{dataset}_epoch{XX}_loss{Y.YYYY}
    # 예: siglip_mlp_anyres-e2p_quic360_epoch03_loss0.4523.ckpt
    # 주의: 이미 위에서 정의된 변수들 재사용 (중복 제거)
    filename_base = f"{vision_short}_{resampler}_{crop_short}_{dataset_name}_epoch{{epoch:02d}}_loss{{val_loss:.4f}}"
    
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=filename_base,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,  # 모델 가중치만 저장 (효율성)
        auto_insert_metric_name=False,
    )
    callbacks.append(ckpt_cb)

    return wandb_logger, callbacks, ckpt_dir


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


def run_stage(
    cfg: Dict[str, Any],
    stage: str,
    stage_manager: StageManager,
    prev_artifact_dir: Optional[str] = None,
    resume_checkpoint_path: Optional[str] = None,
) -> StageResult:
    logger.info(f"=== RUN STAGE: {stage} ===")

    stage_def = stage_manager.get_stage_definition(stage)
    sdef = stage_def.config
    logger.info(f"[STAGE DEFAULTS] {sdef}")

    stage_ip = _resolve_stage_image_processing(cfg, sdef)
    stage_train_data, stage_val_data = _resolve_stage_data(cfg, sdef)
    _save_stage_snapshot(cfg, stage, sdef, stage_ip, stage_train_data, stage_val_data)

    runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
    # experiment.name이 없거나 auto면 구성요소로 생성
    crop = stage_ip.get("crop_strategy", "e2p")
    experiment_name = _compute_experiment_name(cfg, crop_strategy=crop)
    crop_short = crop.replace("_", "-")
    resampler = (
        cfg.get("models", {}).get("resampler_type")
        or cfg.get("models", {}).get("resampler", "mlp")
    )
    # 새로운 디렉토리 구조: {experiment_name}/{stage}/{crop_strategy}_{resampler}
    ckpt_dir = f"{runs_dir}/{experiment_name}/{stage}/{crop_short}_{resampler}"

    dm = build_datamodule(cfg, sdef)
    lit_model = build_model(cfg, stage, sdef, pretrained_dir_override=prev_artifact_dir)

    def _select_resume_checkpoint(candidate: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        path = Path(candidate)
        if path.is_file() and path.suffix == ".ckpt":
            return str(path.resolve())
        if path.is_dir():
            for name in ("last.ckpt", "best.ckpt"):
                resolved = path / name
                if resolved.exists():
                    return str(resolved.resolve())
            ckpt_files = sorted(
                path.glob("*.ckpt"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if ckpt_files:
                return str(ckpt_files[0].resolve())
        logger.warning(f"Resume checkpoint provided but no .ckpt file found: {candidate}")
        return None

    resume_ckpt = _select_resume_checkpoint(resume_checkpoint_path)
    if resume_checkpoint_path and not resume_ckpt:
        logger.warning(f"Unable to resolve resume checkpoint for stage '{stage}': {resume_checkpoint_path}")
    if resume_ckpt:
        logger.info(f"Resuming stage '{stage}' from checkpoint: {resume_ckpt}")

    wandb_logger, callbacks, ckpt_dir = build_logger_and_callbacks(cfg, stage, sdef, dm, lit_model)

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
        precision="bf16-mixed",  # BFloat16 mixed precision for better stability
        gradient_clip_val=1.0,
        default_root_dir=ckpt_dir,
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
        fit_kwargs = {"datamodule": dm}
        if resume_ckpt:
            fit_kwargs["ckpt_path"] = resume_ckpt
        trainer.fit(lit_model, **fit_kwargs)
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
        # LoRA 가중치 추가 저장 (HuggingFace 호환 형식)
        # 주의: Lightning 체크포인트(.ckpt)가 이미 LoRA 포함 state_dict를 저장함
        # 이 별도 저장은 HuggingFace PEFT 라이브러리와의 호환성을 위한 것
        try:
            lora_dir = str(Path(ckpt_dir) / "lora_weights")
            success = lit_model.model.save_lora_weights(lora_dir)
            if success:
                logger.info(f"✓ LoRA weights (HF PEFT format) saved: {lora_dir}")
                logger.info("⚠️  Lightning checkpoint (.ckpt) already contains full LoRA state_dict")
                logger.info("   → Use .ckpt for PanoramaVLM.from_checkpoint() (recommended)")
                logger.info("   → Use lora_weights/ for HuggingFace PEFT compatibility only")
            else:
                logger.warning("⚠️ LoRA weight save returned False")
        except Exception as err:
            logger.warning(f"⚠️ Additional LoRA save failed: {err}")
            logger.info("   Lightning checkpoint still contains full model state")

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
        artifact_dir=ckpt_dir,
        elapsed_minutes=elapsed_minutes,
        error=str(stage_exception) if stage_exception else None,
    )

    if stage_exception is not None:
        raise StageExecutionError(stage, stage_exception, result) from stage_exception

    return result


class StageOrchestrator:
    STATE_VERSION = 1

    def __init__(self, cfg: Dict[str, Any], stage_manager: StageManager):
        self.cfg = cfg
        self.stage_manager = stage_manager
        self.stages = stage_manager.available_stage_names()
        runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
        # experiment.name 우선, fallback으로 training.prefix 사용 (하위 호환성)
        experiment_name = cfg.get("experiment", {}).get("name") or cfg.get("training", {}).get("prefix") or "panovlm_exp"
        self.state_path = Path(runs_dir) / f"{experiment_name}_stage_state.json"
        self.state = self._load_state()
        self.state.setdefault("version", self.STATE_VERSION)
        self.state.setdefault("stages", {})
        force_env = os.environ.get("PANOVLM_FORCE_STAGES", "")
        self.force_stages = {s.strip() for s in force_env.split(",") if s.strip()}
        if self.force_stages:
            logger.info(f"Force rerun stages: {sorted(self.force_stages)}")
        logger.info(f"Stage state file: {self.state_path}")

    def _auto_eval_config(self) -> Dict[str, Any]:
        training = self.cfg.get("training", {}) or {}
        auto_eval = training.get("auto_eval")
        if isinstance(auto_eval, bool):
            return {"enabled": auto_eval}
        if isinstance(auto_eval, dict):
            return dict(auto_eval)
        return {}

    @staticmethod
    def _first_path(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return str(value)

    def _resolve_auto_eval_csv(self, settings: Dict[str, Any]) -> Optional[str]:
        for key in ("csv",):
            if key in settings:
                path = self._first_path(settings[key])
                if path:
                    return path
        paths_cfg = self.cfg.get("paths", {}) or {}
        data_cfg = self.cfg.get("data", {}) or {}
        for candidate in (
            paths_cfg.get("csv_test"),
            data_cfg.get("csv_test"),
        ):
            path = self._first_path(candidate)
            if path:
                return path
        return None

    @staticmethod
    def _select_auto_eval_checkpoint(result: StageResult, preference: str) -> Optional[str]:
        pref = (preference or "last").lower()
        if pref == "best":
            return result.best_checkpoint or result.last_checkpoint or result.artifact_dir
        if pref == "artifact":
            return result.artifact_dir or result.best_checkpoint or result.last_checkpoint
        # default last/current
        return result.last_checkpoint or result.best_checkpoint or result.artifact_dir

    def _maybe_run_auto_eval(self, stage: str, result: StageResult) -> None:
        settings = self._auto_eval_config()
        if not settings.get("enabled"):
            return

        target_stage = settings.get("stage") or "finetune"
        try:
            canonical_target = canonical_stage_name(target_stage)
        except Exception:
            canonical_target = target_stage
        try:
            canonical_current = canonical_stage_name(stage)
        except Exception:
            canonical_current = stage

        if canonical_current != canonical_target:
            return

        checkpoint_path = self._select_auto_eval_checkpoint(result, settings.get("checkpoint", "last"))
        if not checkpoint_path:
            logger.warning("Auto-eval enabled but no checkpoint found after stage '%s'. Skipping.", stage)
            return

        csv_input = self._resolve_auto_eval_csv(settings)
        if not csv_input:
            logger.warning(
                "Auto-eval enabled but no CSV test file provided (set training.auto_eval.csv or paths.csv_test)."
            )
            return

        if not EVAL_SCRIPT_PATH.exists():
            logger.warning("Auto-eval requested but eval.py was not found at %s", EVAL_SCRIPT_PATH)
            return

        cfg_path = self.cfg.get("_config_path")
        cmd = [
            sys.executable,
            str(EVAL_SCRIPT_PATH),
            "--checkpoint",
            str(checkpoint_path),
            "--csv-input",
            str(csv_input),
        ]
        if cfg_path:
            cmd += ["--config", cfg_path]

        if settings.get("metrics_only"):
            cmd.append("--metrics-only")
        if settings.get("log_samples"):
            cmd.append("--log-samples")

        optional_numeric_args = {
            "max_samples": "--max-samples",
            "log_interval": "--log-interval",
            "log_max_samples": "--log-max-samples",
        }
        for key, flag in optional_numeric_args.items():
            value = settings.get(key)
            if value is not None:
                cmd += [flag, str(value)]

        logger.info("🔁 Auto-evaluating stage '%s' with command: %s", stage, " ".join(cmd))
        env = os.environ.copy()
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
        except subprocess.CalledProcessError as exc:
            logger.error("Auto evaluation failed (exit=%s). Command: %s", exc.returncode, " ".join(cmd))
        except FileNotFoundError as exc:
            logger.error("Auto evaluation failed because python executable was not found: %s", exc)

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

    def _cleanup_state_if_completed(self) -> None:
        """Remove stage state file if every planned stage finished successfully."""
        try:
            if not self.state_path.exists():
                return
            stages_state = self.state.get("stages", {})
            for stage in self.stages:
                if stages_state.get(stage, {}).get("status") != "completed":
                    return
            self.state_path.unlink()
            logger.info(f"Removed stage state file: {self.state_path}")
        except Exception as exc:
            logger.warning(f"Failed to remove stage state file ({self.state_path}): {exc}")

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
            upstream_checkpoint = prev_artifact

            # resume_candidate는 stage별 재시작 후보이며, 최초 실행 시에는 upstream과 동일할 수 있음
            resume_checkpoint = None
            if resume_candidate:
                if upstream_checkpoint is None:
                    resume_checkpoint = resume_candidate
                elif os.path.normpath(str(resume_candidate)) != os.path.normpath(str(upstream_checkpoint)):
                    resume_checkpoint = resume_candidate

            init_checkpoint = resume_candidate or upstream_checkpoint

            self._record_stage(
                stage,
                status="running",
                started_at=self._now_iso(),
                error=None,
                upstream_checkpoint=upstream_checkpoint,
                resume_checkpoint=resume_checkpoint,
            )

            try:
                result = run_stage(
                    self.cfg,
                    stage,
                    self.stage_manager,
                    prev_artifact_dir=init_checkpoint,
                    resume_checkpoint_path=resume_checkpoint,
                )
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
                self._maybe_run_auto_eval(stage, result)
                prev_artifact = result.get_load_path() or prev_artifact

        self._cleanup_state_if_completed()
        return prev_artifact


def run_all(cfg: Dict[str, Any], stage_manager: StageManager) -> Optional[str]:
    orchestrator = StageOrchestrator(cfg, stage_manager)
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
        default=None,
        help="(Optional) Comma-separated stage names or 1-based indices to override config.yaml stages.",
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
    bundle = load_runtime_config(args.config)
    cfg = bundle.raw
    cfg["_pano_config_obj"] = bundle.pano
    cfg["_model_config_obj"] = bundle.model
    stage_manager = StageManager(cfg)

    if args.stage:
        try:
            stage_override = stage_manager.resolve_stage_override(args.stage)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)
        if stage_override:
            cfg["_cli_stage_override"] = stage_override
            stage_manager = StageManager(cfg)
            logger.info(f"CLI stage override → {stage_override}")
    else:
        logger.info(f"Using stages from config: {stage_manager.available_stage_names()}")

    if getattr(args, "preview", False):
        _preview_stage_configs(stage_manager)
        sys.exit(0)
    try:
        run_all(cfg, stage_manager)
    except StageExecutionError as err:
        logger.error(f"Stage orchestration failed: {err}")
        sys.exit(1)
