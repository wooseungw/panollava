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
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import default_data_collator
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from contextlib import contextmanager
import gc
import psutil
import time
import json
from typing import Dict, Any, Optional, List, Union

# ── 내부 모듈 ---------------------------------------------------------------
from panovlm.processors.image          import PanoramaImageProcessor
from panovlm.processors.text           import TextTokenizer
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.dataset                   import ChatPanoDataset,  ChatPanoTestDataset
from panovlm.model                     import PanoramaVLM
# ----------------------------------------------------------------------------

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# 전역 설정
class Config:
    """Training configuration management"""
    
    STAGE_DEFAULTS = {
        "vision": {
            "epochs": 1, 
            "lr": 5e-6, 
            "batch_size": 8,   # 32에서 8로 감소
            "vicreg_loss_weight": 1.0, 
            "max_txt_len": 32
        },
        "resampler": {
            "epochs": 1, 
            "lr": 2e-6, 
            "batch_size": 4,   # 16에서 4로 감소
            "vicreg_loss_weight": 0.0, 
            "max_txt_len": 64
        },
        "finetune": {
            "epochs": 1, 
            "lr": 2e-6, 
            "batch_size": 4,   # 16에서 4로 감소
            "vicreg_loss_weight": 0.0, 
            "max_txt_len": 128
        }
    }
    
    MEMORY_THRESHOLDS = {
        "warning": 0.85,  # 85% 메모리 사용 시 경고
        "critical": 0.95  # 95% 메모리 사용 시 긴급 처리
    }

@contextmanager
def memory_monitor():
    """메모리 사용량 모니터링 컨텍스트"""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB (Δ{end_memory-start_memory:+.1f}MB)")

def check_memory_usage():
    """메모리 사용량 체크 및 필요시 가비지 컬렉션"""
    memory_percent = psutil.virtual_memory().percent / 100
    
    if memory_percent > Config.MEMORY_THRESHOLDS["critical"]:
        logger.warning(f"Critical memory usage: {memory_percent:.1%}. Forcing garbage collection.")
        gc.collect(generation=2)  # 가장 접근하기 어려운 객체까지 수집
        torch.cuda.empty_cache()
    elif memory_percent > Config.MEMORY_THRESHOLDS["warning"]:
        logger.warning(f"High memory usage: {memory_percent:.1%}")

def safe_load_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """안전한 체크포인트 로딩"""
    try:
        if str(checkpoint_path).endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(checkpoint_path)
                logger.info(f"Successfully loaded safetensors: {checkpoint_path}")
                return {"state_dict": state_dict}
            except ImportError:
                logger.error("safetensors package not installed. Install with: pip install safetensors")
                return None
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
            return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None

def save_checkpoint_safely(model_state: Dict[str, Any], path: Union[str, Path]):
    """안전한 체크포인트 저장"""
    try:
        # 디렉토리 생성
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # safetensors 우선 시도
        try:
            from safetensors.torch import save_file as save_safetensors
            safetensor_path = str(path).replace(".ckpt", ".safetensors")
            save_safetensors(model_state, safetensor_path)
            logger.info(f"Model saved as safetensors: {safetensor_path}")
        except ImportError:
            torch.save(model_state, path)
            logger.info(f"Model saved as pytorch checkpoint: {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def get_gpu_memory_info():
    """가능한 경우 GPU 메모리 정보 반환"""
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            cached_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            free_memory = total_memory - cached_memory
            return {
                "total": total_memory,
                "allocated": allocated_memory, 
                "cached": cached_memory,
                "free": free_memory,
                "utilization": (cached_memory / total_memory) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None
    return None

def auto_adjust_batch_size(initial_batch_size: int, available_memory_gb: float) -> int:
    """사용 가능한 메모리에 따라 배치 크기 자동 조정"""
    # GPU 메모리 정보 추가 고려
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        logger.info(f"GPU Memory - Total: {gpu_info['total']:.1f}GB, Free: {gpu_info['free']:.1f}GB, Utilization: {gpu_info['utilization']:.1f}%")
        
        # GPU 메모리 기반 조정 (더 엄격한 기준)
        if gpu_info['free'] < 2:
            return max(1, initial_batch_size // 8)
        elif gpu_info['free'] < 4:
            return max(1, initial_batch_size // 4)
        elif gpu_info['free'] < 8:
            return max(1, initial_batch_size // 2)
    
    # RAM 기반 조정 (기존 로직)
    if available_memory_gb < 8:
        return max(1, initial_batch_size // 4)
    elif available_memory_gb < 16:
        return max(1, initial_batch_size // 2)
    else:
        return initial_batch_size

# =============================================================================
# 1. DataModule
# =============================================================================
class VLMDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, batch_size=4, num_workers=4,
                 image_size=(224,224), crop_strategy="e2p",
                 tokenizer_name="Qwen/Qwen3-0.6B", max_txt_len=512, 
                 eval_mode=False):
        super().__init__()
        self.save_hyperparameters()
        
        # 메모리 기반 배치 크기 자동 조정
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        self.hparams.batch_size = auto_adjust_batch_size(batch_size, available_memory)
        
        # 배치 크기 정보 출력
        logger.info(f"=== BATCH SIZE CONFIGURATION ===")
        logger.info(f"Original batch size: {batch_size}")
        logger.info(f"Adjusted batch size: {self.hparams.batch_size}")
        logger.info(f"Available RAM: {available_memory:.1f}GB")
        
        # GPU 메모리 정보 출력
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(f"GPU Memory: {gpu_info['free']:.1f}GB free / {gpu_info['total']:.1f}GB total")
        logger.info(f"================================")
        
        if self.hparams.batch_size != batch_size:
            logger.warning(f"BATCH SIZE ADJUSTED: {batch_size} -> {self.hparams.batch_size} (Available memory: {available_memory:.1f}GB)")
        
        try:
            img_proc = PanoramaImageProcessor(image_size=image_size,
                                              crop_strategy=crop_strategy)
            txt_tok  = TextTokenizer(tokenizer_name, max_len=max_txt_len,)
            self.processor = PanoLLaVAProcessor(img_proc, txt_tok, max_length=512)
            self.tokenizer = txt_tok.tok
            logger.info(f"Data processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data processors: {e}")
            raise
    
    @staticmethod
    def custom_collate_fn(batch):
        # 텐서/배열은 default_data_collator로 처리
        tensor_batch = default_data_collator([{k:v for k,v in item.items() if not isinstance(v,str)} for item in batch])
        # string은 리스트로 따로 모음
        tensor_batch["input_text"] = [item["input_text"] for item in batch]
        return tensor_batch
    
    def setup(self, stage=None):
        try:
            with memory_monitor():
                if self.hparams.eval_mode:
                    # Evaluation 모드: validation 데이터만 로드하고, annotation이 있는 ChatPanoDataset 사용
                    self.val_ds = ChatPanoTestDataset(self.hparams.csv_val,
                                                  self.processor, self.tokenizer)
                    logger.info(f"Evaluation dataset loaded - Val: {len(self.val_ds)}")
                    # Training dataset은 None으로 설정 (evaluation에서는 사용하지 않음)
                    self.train_ds = None
                else:
                    # Training 모드: 정상적인 학습 데이터셋 로드
                    self.train_ds = ChatPanoDataset(self.hparams.csv_train,
                                                    self.processor, self.tokenizer)
                    self.val_ds   = ChatPanoDataset(self.hparams.csv_val,
                                                    self.processor, self.tokenizer)
                    logger.info(f"Training datasets loaded - Train: {len(self.train_ds)}, Val: {len(self.val_ds)}")
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size,
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            collate_fn=self.custom_collate_fn, 
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size,
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            collate_fn=self.custom_collate_fn, 
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None
        )

# =============================================================================
# 2. LightningModule
# =============================================================================
class VLMModule(pl.LightningModule):
    # stage 매핑: 내부적으로 허용되는 값으로 변환
    _STAGE_MAP = {"vision": "vision", "resampler": "resampler", "finetune": "finetune", "generate": "generate"}

    def __init__(self, vision_name, lm_name, resampler, stage, lr):
        super().__init__()
        self.save_hyperparameters()
        self.oom_count = 0  # OOM 발생 횟수 추적
        self.last_oom_step = -1  # 마지막 OOM 발생 스텝
        
        # VICReg loss weight 설정
        vicreg_weight = getattr(self.hparams, 'vicreg_loss_weight', 1.0) if hasattr(self, 'hparams') else 1.0
        self.model = PanoramaVLM(
            vision_model_name=vision_name,
            language_model_name=lm_name,
            resampler_type=resampler,
            vicreg_loss_weight=vicreg_weight
        )
        # stage가 허용되지 않은 값이면 에러
        if stage not in self._STAGE_MAP:
            raise ValueError(f"stage는 {list(self._STAGE_MAP.keys())} 중 하나여야 합니다")
        # stage 매핑
        mapped_stage = self._STAGE_MAP.get(stage, stage)
        self._stage_key = mapped_stage
        self._freeze_for_stage(stage)

    def _freeze_for_stage(self, stage):
        # 전부 잠그고
        self.model.requires_grad_(False)

        if stage == "vision":
            # Stage 1: Vision encoder만 학습 (VICReg loss)
            self.model.vision_encoder.requires_grad_(True)

        elif stage == "resampler":
            # Stage 2: Vision encoder + Resampler + Projection 학습 (VICReg + AR loss)
            self.model.vision_encoder.requires_grad_(True)
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)

        elif stage == "finetune":
            # Stage 3: 전체 모델 학습 (AR loss만)
            self.model.vision_encoder.requires_grad_(True)
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            # Language model도 학습 (특정 레이어만 또는 전체)
            for param in self.model.language_model.parameters():
                param.requires_grad = True

    def forward(self, **batch):
        return self.model(stage=self._stage_key, **batch)


    def training_step(self, batch, _):
        # 메모리 사용량 체크
        if self.global_step % 50 == 0:  # 50스텝마다 체크
            check_memory_usage()
            
            # 현재 배치 크기 및 GPU 메모리 상태 출력
            current_batch_size = batch["pixel_values"].shape[0] if "pixel_values" in batch else "unknown"
            logger.info(f"[Step {self.global_step}] Current batch size: {current_batch_size}")
            
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                logger.info(f"[Step {self.global_step}] GPU Memory: {gpu_info['allocated']:.1f}GB allocated, {gpu_info['free']:.1f}GB free")
        
        try:
            out = self(**batch)
            loss = out["loss"]
            
            # NaN/Inf 체크
            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss detected at step {self.global_step}: {loss}")
                return None
            
            # 로깅
            self.log("loss", loss, prog_bar=True, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
            
            # 단계별 추가 로깅
            if "vicreg_loss" in out:
                self.log("train_vicreg_loss", out["vicreg_loss"], prog_bar=True, sync_dist=True)
            if "ar_loss" in out:
                self.log("train_ar_loss", out["ar_loss"], prog_bar=True, sync_dist=True)
            
            if self.trainer.logger is not None:
                self.trainer.logger.log_metrics({
                    "train_loss": loss.item(),
                    "learning_rate": self.trainer.optimizers[0].param_groups[0]['lr']
                }, step=self.global_step)
            
            return loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.oom_count += 1
                self.last_oom_step = self.global_step
                current_batch_size = batch["pixel_values"].shape[0] if "pixel_values" in batch else "unknown"
                
                logger.error(f"CUDA OOM at step {self.global_step} (OOM #{self.oom_count})")
                logger.error(f"Current batch size: {current_batch_size}")
                logger.error(f"Error: {str(e)}")
                
                # GPU 메모리 정리 시도
                torch.cuda.empty_cache()
                gc.collect()
                
                # 연속적인 OOM 발생 시 경고
                if self.oom_count > 10:
                    logger.error(f"Too many OOMs ({self.oom_count}). Consider reducing batch size manually.")
                    logger.error(f"Current DataLoader batch size: {self.trainer.datamodule.hparams.batch_size}")
                    raise RuntimeError(f"Training stopped due to repeated OOM errors. Total OOMs: {self.oom_count}")
                
                return None
            else:
                logger.error(f"Error in training step {self.global_step}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in training step {self.global_step}: {e}")
            return None

    def validation_step(self, batch, _):
        try:
            out = self(**batch)
            loss = out["loss"]
            
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite validation loss: {loss}")
                return None
            
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            
            if "vicreg_loss" in out:
                self.log("val_vicreg_loss", out["vicreg_loss"], prog_bar=True, sync_dist=True)
            if "ar_loss" in out:
                self.log("val_ar_loss", out["ar_loss"], prog_bar=True, sync_dist=True)
            
            return loss
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            return None

    def configure_optimizers(self):
        # 학습 가능한 파라미터 개수 확인
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params:.1%})")
        
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.hparams.lr, 
            betas=(0.9, 0.98), 
            weight_decay=0.05,
            eps=1e-8
        )
        
        # 스케줄러 설정
        try:
            from transformers import get_linear_schedule_with_warmup
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_steps = steps_per_epoch * self.trainer.max_epochs
            warmup_steps = int(0.1 * total_steps)
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps
            )
            
            logger.info(f"Scheduler configured: {warmup_steps} warmup steps, {total_steps} total steps")
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        except Exception as e:
            logger.warning(f"Failed to configure scheduler: {e}. Using optimizer only.")
            return optimizer

# =============================================================================
# 3. 샘플 로깅 콜백
# =============================================================================
class BatchSizeMonitorCallback(pl.Callback):
    """배치 크기 및 메모리 사용량 모니터링 콜백"""
    
    def on_train_start(self, trainer, pl_module):
        # 훈련 시작 시 배치 크기 정보 출력
        logger.info(f"=== TRAINING START INFO ===")
        logger.info(f"DataLoader batch size: {trainer.datamodule.hparams.batch_size}")
        logger.info(f"Number of training batches: {len(trainer.datamodule.train_dataloader())}")
        logger.info(f"Number of validation batches: {len(trainer.datamodule.val_dataloader())}")
        
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(f"GPU Memory at start: {gpu_info['free']:.1f}GB free / {gpu_info['total']:.1f}GB total")
        logger.info(f"==========================")
    
    def on_train_epoch_start(self, trainer, pl_module):
        # 각 에폭 시작 시 메모리 상태 출력
        logger.info(f"[Epoch {trainer.current_epoch}] Starting training epoch")
        logger.info(f"[Epoch {trainer.current_epoch}] Batch size: {trainer.datamodule.hparams.batch_size}")
        
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(f"[Epoch {trainer.current_epoch}] GPU Memory: {gpu_info['allocated']:.1f}GB allocated, {gpu_info['free']:.1f}GB free")
        
        # OOM 통계 출력
        if hasattr(pl_module, 'oom_count') and pl_module.oom_count > 0:
            logger.warning(f"[Epoch {trainer.current_epoch}] Total OOMs so far: {pl_module.oom_count}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        # 에폭 종료 시 OOM 통계 출력
        if hasattr(pl_module, 'oom_count') and pl_module.oom_count > 0:
            logger.warning(f"[Epoch {trainer.current_epoch}] Epoch ended with {pl_module.oom_count} total OOMs")

class LogSamplesCallback(pl.Callback):
    def __init__(self, tokenizer, num_samples=5, max_new_tokens=32):
        self.tok, self.n, self.m = tokenizer, num_samples, max_new_tokens
        self.last_logged_epoch = -1
    
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # 중복 로깅 방지
        if trainer.current_epoch == self.last_logged_epoch:
            return
        self.last_logged_epoch = trainer.current_epoch
        
        try:
            # 배치 가져오기
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            batch = {k: v.to(pl_module.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            pixel = batch["pixel_values"]
            input_ids = batch.get("input_ids", None)
            image_paths = batch.get("image_path", None)
            
            actual_n = min(self.n, pixel.shape[0])
            if actual_n == 0:
                logger.warning("Validation batch is empty, skipping sample logging.")
                return
            
            # 생성 모드로 추론
            pl_module.eval()
            out = pl_module.model.generate(
                pixel_values=pixel[:actual_n], 
                max_new_tokens=self.m, 
                temperature=0.7
            )
            preds = out["text"]
            
            # 입력 텍스트 디코딩
            input_texts = None
            if input_ids is not None:
                input_texts = self.tok.batch_decode(input_ids[:actual_n], skip_special_tokens=True)
            
            if len(preds) < actual_n:
                logger.warning(f"Model returned fewer predictions ({len(preds)}) than requested ({actual_n})")
            
            # WandB 테이블 생성
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                tbl = wandb.Table(columns=["idx", "image", "image_path", "input_text", "pred"])
                for i in range(min(actual_n, len(preds))):
                    img = pixel[i]
                    if img.dim() == 4:
                        img = img[0]  # (B, 3, H, W) -> (3, H, W)
                    
                    input_str = input_texts[i] if input_texts is not None else "<no input>"
                    pred_str = preds[i] if i < len(preds) else "<no prediction>"
                    img_path = image_paths[i] if image_paths is not None else "<no path>"
                    
                    tbl.add_data(i, wandb.Image(img.cpu()), img_path, input_str, pred_str)
                
                trainer.logger.experiment.log({"val_samples": tbl}, commit=False)
                logger.info(f"Logged {min(actual_n, len(preds))} validation samples")
        
        except Exception as e:
            logger.error(f"Error in sample logging: {e}")
        finally:
            pl_module.train()

# =============================================================================
# 4. main
# =============================================================================

def run_stages(args, stages=None, prev_ckpt=None):
    """
    Multi-stage training orchestrator
    
    Args:
        args: Training arguments
        stages: None (single stage), str (specific stage), or list/tuple (multiple stages)
        prev_ckpt: Previous checkpoint path
    
    Returns:
        Path to the best checkpoint from the last stage
    """
    start_time = time.time()
    
    try:
        if stages is None:
            # 단일 스테이지
            logger.info(f"Starting single stage training: {args.stage}")
            prev_ckpt = _run_stage_core(args, args.stage, prev_ckpt=args.resume_from if args.resume_from else None)
            logger.info(f"Stage {args.stage} completed. Best checkpoint: {prev_ckpt}")
            return prev_ckpt
        
        elif isinstance(stages, str):
            # 지정 스테이지 하나만 학습
            args.stage = stages
            logger.info(f"Starting specific stage training: {stages}")
            prev_ckpt = _run_stage_core(args, stages, prev_ckpt=args.resume_from if args.resume_from else None)
            logger.info(f"Stage {stages} completed. Best checkpoint: {prev_ckpt}")
            return prev_ckpt
        
        elif isinstance(stages, (list, tuple)):
            # 여러 스테이지 반복
            logger.info(f"Starting multi-stage training: {stages}")
            for i, stage in enumerate(stages):
                args.stage = stage
                logger.info(f"Starting stage {i+1}/{len(stages)}: {stage}")
                
                with memory_monitor():
                    prev_ckpt = _run_stage_core(args, stage, prev_ckpt)
                
                logger.info(f"Stage {stage} completed. Best checkpoint: {prev_ckpt}")
                
                # 중간 메모리 정리
                if i < len(stages) - 1:  # 마지막 스테이지가 아닐 때만
                    gc.collect()
                    torch.cuda.empty_cache()
            
            return prev_ckpt
        
        else:
            raise ValueError("stages는 None, str, list/tuple 중 하나여야 합니다.")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Total training time: {elapsed_time/3600:.1f} hours")

def _run_stage_core(args, stage, prev_ckpt=None):
    """
    Core training function for a single stage
    
    Args:
        args: Training arguments
        stage: Training stage ('vision', 'resampler', 'finetune')
        prev_ckpt: Previous checkpoint path
    
    Returns:
        Path to the best checkpoint
    """
    logger.info(f"Configuring stage: {stage}")
    
    # 스테이지별 기본값 적용
    stage_hparams = Config.STAGE_DEFAULTS.get(stage, {})
    original_values = {}
    
    for k, v in stage_hparams.items():
        attr_name = k.replace('-', '_')  # hyphen to underscore
        cur = getattr(args, attr_name, None)
        
        # 기본값 적용 조건
        if cur is None or (isinstance(v, int) and cur == 0) or (isinstance(v, float) and cur == 0.0):
            original_values[attr_name] = cur
            setattr(args, attr_name, v)
            logger.info(f"Applied stage default {attr_name}: {cur} -> {v}")
    
    # 데이터 모듈 초기화
    try:
        dm = VLMDataModule(
            csv_train=args.csv_train,
            csv_val=args.csv_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer_name=args.lm_name,
            max_txt_len=args.max_txt_len,
            crop_strategy=args.crop_strategy,
        )
    except Exception as e:
        logger.error(f"Failed to initialize data module: {e}")
        raise
    
    # 체크포인트 로딩 및 스테이지 변경 감지
    is_stage_change = False
    checkpoint = None
    if prev_ckpt:
        checkpoint = safe_load_checkpoint(prev_ckpt)
        if checkpoint:
            prev_stage = checkpoint.get('hyper_parameters', {}).get('stage')
            if prev_stage and prev_stage != stage:
                is_stage_change = True
                logger.info(f"Stage changed ({prev_stage} → {stage}): Loading weights only")
        else:
            logger.warning("Failed to load checkpoint, assuming stage change")
            is_stage_change = True
    
    # 모델 초기화
    try:
        lit_model = VLMModule(args.vision_name, args.lm_name, args.resampler, stage, args.lr)
        
        # 체크포인트에서 가중치 로드
        if prev_ckpt and checkpoint:
            state_dict = checkpoint.get("state_dict", checkpoint)
            if state_dict:
                missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if missing:
                    logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
                if unexpected:
                    logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    
    # 실행 설정
    run_name = f"{stage}_{Path(args.csv_train).stem}_{int(time.time())}"
    wandb_dir = "./runs"
    Path(wandb_dir).mkdir(exist_ok=True)
    
    # WandB 설정
    wandb_config = {
        "stage": stage,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "vicreg_loss_weight": getattr(args, "vicreg_loss_weight", None),
        "vision_name": args.vision_name,
        "lm_name": args.lm_name,
        "resampler": args.resampler,
        "max_txt_len": args.max_txt_len,
        "csv_train": args.csv_train,
        "csv_val": args.csv_val,
        "num_workers": args.num_workers,
        "crop_strategy": args.crop_strategy,
        "image_size": args.image_size,
    }
    
    # 콜백 설정
    callbacks = []
    
    # 배치 크기 모니터링 콜백 (항상 추가)
    callbacks.append(BatchSizeMonitorCallback())
    
    # 체크포인트 콜백
    ckpt_dir = f"./runs/{args.crop_strategy}_{stage}_{args.resampler}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
        dirpath=ckpt_dir,
        verbose=True
    )
    callbacks.append(ckpt_cb)
    
    # 샘플 로깅 콜백
    if stage in ["resampler", "finetune"]:
        callbacks.append(LogSamplesCallback(dm.tokenizer))
    
    # Early stopping 콜백
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=False  # step 단위로 체크
    )
    callbacks.append(early_stop_cb)
    
    # 로거 초기화
    try:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            dir=wandb_dir,
            save_dir=wandb_dir
        )
    except Exception as e:
        logger.warning(f"Failed to initialize WandB logger: {e}. Training will continue without WandB.")
        wandb_logger = None
    
    # 트레이너 초기화
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval = 0.25,
        max_epochs=args.epochs,
        precision="16-mixed",
        gradient_clip_val=0.5,
        accelerator="auto",
        default_root_dir=ckpt_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True
    )
    
    # 훈련 시작
    try:
        logger.info(f"Starting training for stage: {stage}")
        start_time = time.time()
        
        if prev_ckpt and not is_stage_change:
            trainer.fit(lit_model, datamodule=dm, ckpt_path=prev_ckpt)
        else:
            trainer.fit(lit_model, datamodule=dm)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time/60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"Training failed for stage {stage}: {e}")
        raise
    
    # 최종 모델 저장 (각 stage별 폴더)
    try:
        final_model_path = str(Path(ckpt_dir) / "model_final.safetensors")
        save_checkpoint_safely(lit_model.state_dict(), final_model_path)
        logger.info(f"Final model saved at: {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")
    
    # 원래 값들 복원
    for attr_name, original_value in original_values.items():
        setattr(args, attr_name, original_value)
    
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
    p.add_argument('--crop-strategy', default='e2p', 
                       choices=['sliding_window', 'e2p', 'cubemap', 'resize', 'anyres', 'anyres_max'],
                       help='Image cropping strategy')
    p.add_argument("--image-size", type=int, nargs=2, default=(224, 224),
                   help="이미지 크기 (예: 224 224)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)  # 64에서 4로 감소
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--vicreg-loss-weight", type=float, default=0.0, help="VICReg loss weight for each stage")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-txt-len", type=int, default=32)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--wandb-project", default="panorama-vlm")
    p.add_argument("--wandb-name",    default=None)
    args = p.parse_args()

    # 단일/전체 스테이지 학습 통합
    if args.stages is not None and len(args.stages) > 0:
        stages = args.stages if isinstance(args.stages, list) else args.stages.split()
        prev_ckpt = args.resume_from if args.resume_from else None
        run_stages(args, stages, prev_ckpt=prev_ckpt)
    else:
        run_stages(args)
