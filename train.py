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
import traceback  # 추가됨
from typing import Dict, Any, Optional, List, Union

# ── 내부 모듈 ---------------------------------------------------------------
# from panovlm.processors.image          import PanoramaImageProcessor
# from panovlm.processors.text           import TextTokenizer
# from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.dataset                   import VLMDataModule
from panovlm.model                     import PanoramaVLM
from panovlm.utils                     import *
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

# =============================================================================
# 2. LightningModule
# =============================================================================
class VLMModule(pl.LightningModule):
    """
    Panorama Vision-Language Model Lightning Module
    
    3단계 학습을 지원하는 파노라마 비전-언어 모델의 PyTorch Lightning 래퍼 클래스.
    각 단계별로 다른 파라미터 그룹을 동결/해제하여 단계적 학습을 수행합니다.
    
    학습 단계:
        - vision: Vision encoder만 학습 (VICReg loss 사용)
        - resampler: Vision encoder + Resampler + Projection 학습 (AR loss)  
        - finetune: 전체 모델 또는 LoRA 어댑터 학습 (AR loss)
    
    Args:
        vision_name (str): Vision encoder 모델명 (HuggingFace model name)
        lm_name (str): Language model 모델명 (HuggingFace model name)
        resampler (str): Resampler 타입 ('mlp' 등)
        stage (str): 학습 단계 ('vision', 'resampler', 'finetune')
        lr (float): 학습률
        max_text_length (int, optional): 최대 텍스트 길이
        vicreg_loss_weight (float, optional): VICReg loss 가중치
        use_lora (bool): LoRA 사용 여부 (finetune 단계에서만 적용)
        lora_rank (int): LoRA rank
        lora_alpha (int): LoRA alpha 파라미터
        lora_dropout (float): LoRA dropout rate
        lora_target_modules (list, optional): LoRA를 적용할 모듈 이름들
        
    Attributes:
        model (PanoramaVLM): 실제 VLM 모델 인스턴스
        oom_count (int): Out-of-Memory 발생 횟수 추적
        last_oom_step (int): 마지막 OOM 발생 스텝
        _stage_key (str): 내부 처리용 단계 키
        use_lora (bool): LoRA 사용 여부 저장
    """
    # stage 매핑: 내부적으로 허용되는 값으로 변환
    _STAGE_MAP = {"vision": "vision", "resampler": "resampler", "finetune": "finetune", "generate": "generate"}

    def __init__(self, 
                 vision_name = "google/siglip-base-patch16-224", 
                 lm_name = "Qwen/Qwen3-0.6B", 
                 resampler = "mlp", 
                 stage = "vision", 
                 lr = 2e-6,
                 max_text_length = None,
                 vicreg_loss_weight = None,  # VICReg loss weight 파라미터 추가
                 overlap_ratio = 0.5,  # VICReg overlap ratio
                 # LoRA 파라미터들
                 use_lora = False,
                 lora_rank = 16,
                 lora_alpha = 32,
                 lora_dropout = 0.1,
                 lora_target_modules = None
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.oom_count = 0  # OOM 발생 횟수 추적
        self.last_oom_step = -1  # 마지막 OOM 발생 스텝
        
        # max_text_length 기본값 설정
        if max_text_length is None:
            max_text_length = 512
        
        # VICReg loss weight 설정
        if vicreg_loss_weight is not None:
            # 명시적으로 전달된 값 사용
            vicreg_weight = vicreg_loss_weight
        else:
            # 스테이지별 기본값: vision stage에서는 1.0, 다른 stage에서는 0.0
            vicreg_weight = 1.0 if stage == "vision" else 0.0
            
        logger.info(f"VICReg loss weight set to: {vicreg_weight} for stage: {stage}")
        
        self.model = PanoramaVLM(
            vision_model_name=vision_name,
            language_model_name=lm_name,
            resampler_type=resampler,
            vicreg_loss_weight=vicreg_weight,
            vicreg_overlap_ratio=overlap_ratio,
            max_text_length=max_text_length
        )
        # stage가 허용되지 않은 값이면 에러
        if stage not in self._STAGE_MAP:
            raise ValueError(f"stage는 {list(self._STAGE_MAP.keys())} 중 하나여야 합니다")
        # stage 매핑
        mapped_stage = self._STAGE_MAP.get(stage, stage)
        self._stage_key = mapped_stage
        self.use_lora = use_lora
        
        # LoRA 설정 (finetune 단계에서만)
        if use_lora and stage == "finetune":
            logger.info("Setting up LoRA for finetune stage...")
            success = self.model.setup_lora_for_finetune(
                lora_r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )
            if success:
                logger.info("✓ LoRA setup completed successfully")
            else:
                logger.warning("⚠ LoRA setup failed, continuing with full finetuning")
        elif use_lora and stage != "finetune":
            logger.warning(f"⚠ LoRA is only supported for finetune stage, but current stage is '{stage}'. Ignoring LoRA settings.")
        
        self._freeze_for_stage(stage)
        
        # 체크포인트 호환성을 위한 추가 메타데이터 저장 준비
        self._prepare_checkpoint_metadata()

    def _freeze_for_stage(self, stage):
        """
        스테이지별 파라미터 동결 설정
        
        각 학습 단계에 따라 적절한 모델 구성요소만 학습 가능하도록 설정합니다.
        단계적 학습을 통해 모델의 안정적인 수렴을 도모합니다.
        
        Args:
            stage (str): 학습 단계
                - 'vision': Vision encoder만 학습 가능
                - 'resampler': Vision encoder + Resampler + Projection 학습 가능
                - 'finetune': 전체 모델 또는 LoRA 어댑터 학습 가능
                
        Processing:
            1. 모든 파라미터를 먼저 동결 (requires_grad=False)
            2. 해당 단계에 필요한 구성요소만 해제 (requires_grad=True)
            3. 학습 가능한 파라미터 수 통계 출력
            
        Side Effects:
            - 모델 파라미터의 requires_grad 속성 변경
            - 콘솔에 설정 정보 및 파라미터 통계 로그 출력
        """
        # 전부 잠그고
        self.model.requires_grad_(False)

        if stage == "vision":
            # Stage 1: Vision encoder만 학습 (VICReg loss)
            self.model.vision_encoder.requires_grad_(True)
            logger.info("✓ Stage 1: Only vision encoder unfrozen")

        elif stage == "resampler":
            # Stage 2: Vision encoder + Resampler + Projection 학습 (VICReg + AR loss)
            self.model.vision_encoder.requires_grad_(True)  # 주석 해제됨
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            logger.info("✓ Stage 2: Vision encoder + Resampler + Projection unfrozen")

        elif stage == "finetune":
            # Stage 3: Projection, LLM(Lora) 모델 학습 (AR loss만), 일단 임시로 모두 학습
            # self.model.vision_encoder.requires_grad_(True)  # 주석 해제됨
            self.model.resampler.requires_grad_(True)
            self.model.vision_to_language_projection.requires_grad_(True)
            
            # LoRA 사용 여부에 따라 Language model 학습 여부 결정
            if not self.use_lora:  # 조건문 개선
                for param in self.model.language_model.parameters():
                    param.requires_grad = True
                logger.info("✓ Stage 3: Full model unfrozen (no LoRA)")
            else:
                # LoRA 사용시 language model은 LoRA 어댑터만 학습
                logger.info("✓ Stage 3: Vision components + LoRA adapters unfrozen")
                
        # 현재 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params:.1%})")
                
    def forward(self, **batch): 
        """
        모델 순전파 실행
        
        배치 데이터를 모델에 전달하여 순전파를 수행합니다.
        현재 설정된 stage에 따라 적절한 모델 처리 방식을 사용합니다.
        
        Args:
            **batch: 배치 데이터 (키워드 인자)
                - pixel_values (torch.Tensor): 이미지 픽셀 값 (B, V, C, H, W) 또는 (B, C, H, W)
                - input_ids (torch.Tensor): 토큰화된 텍스트 입력 (B, L)
                - attention_mask (torch.Tensor): 어텐션 마스크 (B, L)
                - labels (torch.Tensor, optional): 학습용 라벨 (B, L)
                
        Returns:
            dict: 모델 출력 딕셔너리
                - loss (torch.Tensor): 계산된 손실값
                - vicreg_loss (torch.Tensor, optional): VICReg 손실 (vision stage)
                - ar_loss (torch.Tensor, optional): Autoregressive 손실 (resampler/finetune stage)
                - logits (torch.Tensor, optional): 모델 로짓 출력
        """
        return self.model(stage=self._stage_key, **batch)

    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning 훈련 스텝 실행
        
        각 훈련 배치에 대해 순전파, 손실 계산, 로깅을 수행합니다.
        OOM(Out-of-Memory) 에러 처리 및 무한/NaN 손실 검증을 포함합니다.
        
        Args:
            batch (dict): 훈련 배치 데이터
                - pixel_values (torch.Tensor): 이미지 데이터
                - input_ids (torch.Tensor): 텍스트 토큰
                - attention_mask (torch.Tensor): 어텐션 마스크
                - labels (torch.Tensor): 학습 라벨
            batch_idx (int): 배치 인덱스
            
        Returns:
            torch.Tensor or None: 계산된 손실값 (에러 시 None 반환)
            
        Processing:
            1. 모델 순전파 실행 (**batch)
            2. 손실값 유효성 검증 (NaN/Inf 체크)
            3. 배치 크기 추출 및 메트릭 로깅
            4. 단계별 추가 손실 로깅 (vicreg_loss, ar_loss)
            5. WandB 로깅 (10 step마다)
            6. 예외 처리 (OOM, 일반 에러)
            
        Side Effects:
            - Lightning 메트릭 로깅 (self.log)
            - WandB 로깅 (trainer.logger)
            - OOM 통계 업데이트 (self.oom_count)
            - 에러 시 콘솔 로그 출력
        """
        try:
            out = self(**batch)
            loss = out["loss"]
            # 명시적 batch_size 전달 (Lightning 경고 방지)
            bs = None
            try:
                if isinstance(batch.get("pixel_values"), torch.Tensor):
                    bs = batch["pixel_values"].size(0)
            except Exception:
                bs = None
            
            # NaN/Inf 체크
            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss detected at step {self.global_step}: {loss}")
                # 현재 배치 정보 로깅
                batch_info = {k: v.shape if torch.is_tensor(v) else len(v) if isinstance(v, list) else type(v) 
                             for k, v in batch.items()}
                logger.error(f"Batch info: {batch_info}")
                return None
            
            # 로깅
            if bs is not None:
                self.log("loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
            else:
                self.log("loss", loss, prog_bar=True, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
            
            # 단계별 추가 로깅
            if "vicreg_loss" in out:
                if bs is not None:
                    self.log("train_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False, batch_size=bs)
                else:
                    self.log("train_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False)
            if "ar_loss" in out:
                if bs is not None:
                    self.log("train_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False, batch_size=bs)
                else:
                    self.log("train_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False)
            
            # WandB 로깅
            if self.trainer.logger is not None and batch_idx % 10 == 0:  # 10스텝마다
                self.trainer.logger.log_metrics({
                    "train_loss": loss.item(),
                    "learning_rate": self.trainer.optimizers[0].param_groups[0]['lr'],
                    "global_step": self.global_step
                }, step=self.global_step)
            
            return loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.oom_count += 1
                self.last_oom_step = self.global_step
                current_batch_size = batch["pixel_values"].shape[0] if "pixel_values" in batch else "unknown"
                
                logger.error(f"CUDA OOM at step {self.global_step} (OOM #{self.oom_count})")
                logger.error(f"Current batch size: {current_batch_size}")
                
                # GPU 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
                
                # 연속적인 OOM 발생 시 경고
                if self.oom_count > 10:
                    logger.error(f"Too many OOMs ({self.oom_count}). Consider reducing batch size.")
                    raise RuntimeError(f"Training stopped due to repeated OOM errors. Total OOMs: {self.oom_count}")
                
                return None
            else:
                logger.error(f"Runtime error in training step {self.global_step}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in training step {self.global_step}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def validation_step(self, batch, batch_idx):
        """
        PyTorch Lightning 검증 스텝 실행
        
        검증 배치에 대해 순전파를 수행하고 손실을 계산합니다.
        훈련 스텝과 유사하지만 역전파는 수행하지 않습니다.
        
        Args:
            batch (dict): 검증 배치 데이터 (training_step과 동일 형태)
            batch_idx (int): 배치 인덱스
            
        Returns:
            torch.Tensor or None: 계산된 검증 손실값 (에러 시 None 반환)
            
        Processing:
            1. torch.no_grad() 하에서 모델 순전파 수행
            2. 손실값 유효성 검증
            3. 검증 메트릭 로깅 (val_ 접두어)
            4. 단계별 검증 손실 로깅
            5. 예외 처리 및 에러 로깅
            
        Side Effects:
            - Lightning 검증 메트릭 로깅
            - 에러 시 콘솔 로그 출력
        """
        try:
            out = self(**batch)
            loss = out["loss"]
            # 명시적 batch_size 전달 (Lightning 경고 방지)
            bs = None
            try:
                if isinstance(batch.get("pixel_values"), torch.Tensor):
                    bs = batch["pixel_values"].size(0)
            except Exception:
                bs = None
            
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite validation loss at step {batch_idx}: {loss}")
                return None
            
            # 로깅
            if bs is not None:
                self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
            else:
                self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            
            if "vicreg_loss" in out:
                if bs is not None:
                    self.log("val_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False, batch_size=bs)
                else:
                    self.log("val_vicreg_loss", out["vicreg_loss"], prog_bar=False, sync_dist=False)
            if "ar_loss" in out:
                if bs is not None:
                    self.log("val_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False, batch_size=bs)
                else:
                    self.log("val_ar_loss", out["ar_loss"], prog_bar=False, sync_dist=False)
                
            return loss
            
        except Exception as e:
            logger.error(f"Error in validation step {batch_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def configure_optimizers(self):
        """
        PyTorch Lightning 옵티마이저 및 스케줄러 설정
        
        학습 가능한 파라미터에 대해 AdamW 옵티마이저와 
        Linear Warmup 스케줄러를 설정합니다.
        
        Returns:
            dict or torch.optim.Optimizer: 옵티마이저 설정
                - optimizer: AdamW 옵티마이저
                - lr_scheduler: Linear warmup 스케줄러 (설정 성공 시)
                
        Processing:
            1. 학습 가능한 파라미터만 필터링하여 옵티마이저 생성
            2. 전체 훈련 스텝 수 계산 (epochs × steps_per_epoch)
            3. Warmup 스텝 수 계산 (전체 스텝의 10%)
            4. Linear warmup 스케줄러 설정
            5. 실패 시 옵티마이저만 반환
            
        Optimizer Settings:
            - AdamW with lr=self.hparams.lr
            - betas=(0.9, 0.98), weight_decay=0.05, eps=1e-8
            
        Scheduler Settings:
            - Linear warmup with 10% of total steps
            - 매 step마다 업데이트 (interval='step')
        """
        # 학습 가능한 파라미터 개수는 _freeze_for_stage에서 이미 출력됨
        
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
            
            logger.info(f"✓ Scheduler configured: {warmup_steps} warmup steps, {total_steps} total steps")
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            
        except Exception as e:
            logger.warning(f"Failed to configure scheduler: {e}. Using optimizer only.")
            return optimizer
    
    def _prepare_checkpoint_metadata(self):
        """
        새로운 from_checkpoint 인터페이스와 호환성을 위한 메타데이터 준비
        """
        # 모델 설정 정보를 hparams에 추가하여 체크포인트에 포함
        model_config = {
            # PanoramaVLM 생성에 필요한 파라미터들
            'vision_model_name': getattr(self.model.vision_encoder.config, 'name_or_path', 'google/siglip-base-patch16-224'),
            'language_model_name': getattr(self.model.language_model.config, 'name_or_path', 'Qwen/Qwen3-0.6B'),
            'resampler_type': 'mlp',  # 현재 하드코딩
            'latent_dimension': self.model.vision_to_language_projection.in_features,
            'vicreg_loss_weight': self.model.vicreg_loss_weight,
            'vicreg_overlap_ratio': self.model.vicreg_overlap_ratio,
            'max_text_length': self.model.max_text_length,
            
            # 훈련 관련 메타데이터
            'stage': self._stage_key,
            'use_lora': self.use_lora,
        }
        
        # LoRA 정보 추가 (있다면)
        if self.use_lora:
            lora_info = self.model.get_lora_info()
            if lora_info.get("is_lora_enabled", False):
                model_config.update({
                    'lora_r': lora_info.get('lora_r'),
                    'lora_alpha': lora_info.get('lora_alpha'),
                    'lora_dropout': lora_info.get('lora_dropout'),
                    'lora_target_modules': lora_info.get('target_modules'),
                })
        
        # hparams에 병합
        for key, value in model_config.items():
            if key not in self.hparams:
                self.hparams[key] = value
        
        logger.info(f"✓ 체크포인트 메타데이터 준비 완료 ({len(model_config)} 항목)")

# =============================================================================
# 3. 샘플 로깅 콜백
# =============================================================================
class BatchSizeMonitorCallback(pl.Callback):
    """
    훈련 모니터링 및 설정 로깅 콜백
    
    PyTorch Lightning 콜백으로서 훈련 시작/종료 시점에
    상세한 설정 정보, 메모리 상태, OOM 통계를 로깅합니다.
    
    주요 기능:
        - 모델, 데이터셋, 훈련 설정의 상세 정보 로깅
        - WandB에 설정 정보 자동 업로드
        - GPU 메모리 상태 모니터링
        - OOM 통계 추적 및 경고
        - 환경 변수 및 하드웨어 정보 로깅
        
    Methods:
        on_train_start: 훈련 시작 시 전체 설정 정보 로깅
        on_train_epoch_start: 에포크 시작 시 메모리 상태 로깅
        on_train_epoch_end: 에포크 종료 시 OOM 통계 로깅
        _log_config_info: 콘솔에 상세 설정 정보 출력
        _get_config_dict: WandB용 설정 딕셔너리 생성
    """
    
    def on_train_start(self, trainer, pl_module):
        """
        훈련 시작 시 실행되는 콜백 메서드
        
        훈련이 시작될 때 모델, 데이터셋, 하드웨어 등의 
        상세한 설정 정보를 콘솔과 WandB에 로깅합니다.
        
        Args:
            trainer (pl.Trainer): PyTorch Lightning 트레이너 인스턴스
            pl_module (VLMModule): 훈련 중인 모델 모듈
            
        Processing:
            1. 상세 설정 정보를 콘솔에 출력 (_log_config_info)
            2. 데이터로더 통계 정보 로깅
            3. WandB 실험 설정에 config 정보 업로드
            
        Side Effects:
            - 콘솔에 상세 로그 출력
            - WandB config 업데이트 (trainer.logger.experiment.config.update)
        """
        logger.info(f"=== TRAINING START INFO ===")
        
        # Config 정보 로깅
        self._log_config_info(trainer, pl_module)
        
        # 데이터로더 정보
        logger.info(f"DataLoader batch size: {trainer.datamodule.hparams.batch_size}")
        logger.info(f"Number of training batches: {len(trainer.datamodule.train_dataloader())}")
        logger.info(f"Number of validation batches: {len(trainer.datamodule.val_dataloader())}")
        
        # WandB에 config 정보 로깅 (allow_val_change=True로 기존 값 변경 허용)
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            config_dict = self._get_config_dict(trainer, pl_module)
            trainer.logger.experiment.config.update(config_dict, allow_val_change=True)
    
    def _log_config_info(self, trainer, pl_module):
        """
        상세한 설정 정보를 콘솔에 로깅
        
        모델, 데이터셋, 훈련, 환경 설정을 구조화된 형태로 출력합니다.
        디버깅 및 실험 추적에 유용한 정보를 제공합니다.
        
        Args:
            trainer (pl.Trainer): PyTorch Lightning 트레이너
            pl_module (VLMModule): 모델 모듈
            
        Output Sections:
            - MODEL CONFIGURATION: 모델 구성 정보
            - DATASET CONFIGURATION: 데이터셋 설정
            - TRAINING CONFIGURATION: 훈련 파라미터
            - ENVIRONMENT INFO: 환경변수 및 하드웨어 정보
            
        Side Effects:
            - logger.info()를 통한 콘솔 출력
        """
        logger.info(f"=== MODEL CONFIGURATION ===")
        logger.info(f"Stage: {pl_module._stage_key}")
        logger.info(f"Vision Model: {getattr(pl_module.model, 'vision_encoder', {}).config.name_or_path if hasattr(getattr(pl_module.model, 'vision_encoder', {}), 'config') else 'Unknown'}")
        logger.info(f"Language Model: {getattr(pl_module.model, 'language_model', {}).config.name_or_path if hasattr(getattr(pl_module.model, 'language_model', {}), 'config') else 'Unknown'}")
        logger.info(f"Resampler Type: {getattr(pl_module.model, 'resampler', {}).__class__.__name__ if hasattr(pl_module.model, 'resampler') else 'Unknown'}")
        logger.info(f"Max Text Length: {getattr(pl_module.model, 'max_text_length', 'Unknown')}")
        logger.info(f"Use LoRA: {pl_module.use_lora}")
        if pl_module.use_lora:
            logger.info(f"LoRA Rank: {getattr(pl_module, 'lora_rank', 'Unknown')}")
            logger.info(f"LoRA Alpha: {getattr(pl_module, 'lora_alpha', 'Unknown')}")
            logger.info(f"LoRA Dropout: {getattr(pl_module, 'lora_dropout', 'Unknown')}")
        
        logger.info(f"=== DATASET CONFIGURATION ===")
        logger.info(f"Train CSV: {trainer.datamodule.hparams.csv_train}")
        logger.info(f"Val CSV: {trainer.datamodule.hparams.csv_val}")
        logger.info(f"Image Size: {trainer.datamodule.hparams.image_size}")
        logger.info(f"Crop Strategy: {trainer.datamodule.hparams.crop_strategy}")
        logger.info(f"System Message: {trainer.datamodule.hparams.system_msg}")
        
        logger.info(f"=== TRAINING CONFIGURATION ===")
        logger.info(f"Batch Size: {trainer.datamodule.hparams.batch_size}")
        logger.info(f"Number of Workers: {trainer.datamodule.hparams.num_workers}")
        logger.info(f"Max Epochs: {trainer.max_epochs}")
        logger.info(f"Learning Rate: {getattr(pl_module, 'lr', 'Unknown')}")
        
        # 환경 변수 정보
        import os
        logger.info(f"=== ENVIRONMENT INFO ===")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        logger.info(f"WANDB_PROJECT: {os.environ.get('WANDB_PROJECT', 'Not Set')}")
        logger.info(f"Python Path: {os.environ.get('PYTHONPATH', 'Not Set')}")
        
        # GPU 정보
        if torch.cuda.is_available():
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            logger.info(f"Current GPU: {torch.cuda.current_device()}")
            logger.info(f"GPU Name: {torch.cuda.get_device_name()}")
            
        logger.info(f"================================")
    
    def _get_config_dict(self, trainer, pl_module):
        """
        WandB 로깅용 설정 딕셔너리 생성
        
        모든 설정 정보를 구조화된 딕셔너리로 변환하여
        WandB 실험 추적에 사용할 수 있도록 준비합니다.
        
        Args:
            trainer (pl.Trainer): PyTorch Lightning 트레이너
            pl_module (VLMModule): 모델 모듈
            
        Returns:
            dict: WandB config용 딕셔너리
                - Model config: 모델 구성 정보
                - Dataset config: 데이터셋 설정
                - Training config: 훈련 파라미터
                - Hardware info: GPU 및 시스템 정보
                - Environment info: 환경변수 및 버전 정보
                
        Processing:
            1. 모델 속성에서 안전하게 정보 추출 (getattr 사용)
            2. 환경변수에서 시스템 정보 수집
            3. GPU 가용성에 따른 하드웨어 정보 추가
            4. Python/PyTorch 버전 정보 포함
        """
        import os
        
        config = {
            # Model config
            "stage": pl_module._stage_key,
            "vision_model": getattr(getattr(pl_module.model, 'vision_encoder', {}), 'name_or_path', 'Unknown'),
            "language_model": getattr(getattr(pl_module.model, 'language_model', {}), 'name_or_path', 'Unknown'),
            "resampler_type": getattr(pl_module.model, 'resampler', {}).__class__.__name__ if hasattr(pl_module.model, 'resampler') else 'Unknown',
            "max_text_length": getattr(pl_module.model, 'max_text_length', None),
            "use_lora": pl_module.use_lora,
            
            # Dataset config
            "train_csv": trainer.datamodule.hparams.csv_train,
            "val_csv": trainer.datamodule.hparams.csv_val,
            "image_size": trainer.datamodule.hparams.image_size,
            "crop_strategy": trainer.datamodule.hparams.crop_strategy,
            "system_msg": trainer.datamodule.hparams.system_msg,
            
            # Training config
            "batch_size": trainer.datamodule.hparams.batch_size,
            "num_workers": trainer.datamodule.hparams.num_workers,
            "max_epochs": trainer.max_epochs,
            "learning_rate": getattr(pl_module, 'lr', None),
            
            # Hardware info
            "num_gpus": trainer.num_devices if hasattr(trainer, 'num_devices') else 1,
            "strategy": trainer.strategy.__class__.__name__ if hasattr(trainer, 'strategy') else 'Unknown',
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'),
            
            # Environment info
            "wandb_project": os.environ.get('WANDB_PROJECT', 'Not Set'),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pytorch_version": torch.__version__,
            "lightning_version": pl.__version__,
        }
        
        # LoRA 설정 (사용할 때만 추가)
        if pl_module.use_lora:
            lora_rank = getattr(pl_module, 'lora_rank', None)
            lora_alpha = getattr(pl_module, 'lora_alpha', None)
            lora_dropout = getattr(pl_module, 'lora_dropout', None)
            
            if lora_rank is not None:
                config["lora_rank"] = lora_rank
            if lora_alpha is not None:
                config["lora_alpha"] = lora_alpha
            if lora_dropout is not None:
                config["lora_dropout"] = lora_dropout
        
        # GPU 정보 추가
        if torch.cuda.is_available():
            config.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(),
                "current_gpu": torch.cuda.current_device(),
            })
            
        return config
    
    def on_train_epoch_start(self, trainer, pl_module):
        """
        훈련 에포크 시작 시 실행되는 콜백 메서드
        
        각 에포크가 시작될 때 메모리 상태를 모니터링하고
        OOM 통계를 출력합니다.
        
        Args:
            trainer (pl.Trainer): PyTorch Lightning 트레이너
            pl_module (VLMModule): 모델 모듈
            
        Processing:
            1. 현재 에포크 번호 로깅
            2. GPU 메모리 사용량 확인 및 로깅
            3. 누적 OOM 횟수 경고 출력
            
        Side Effects:
            - 에포크 시작 로그 출력
            - GPU 메모리 상태 로깅
            - OOM 경고 메시지 (발생한 경우)
        """
        logger.info(f"[Epoch {trainer.current_epoch}] Starting training epoch")
        
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(f"[Epoch {trainer.current_epoch}] GPU Memory: {gpu_info['allocated']:.1f}GB allocated, {gpu_info['free']:.1f}GB free")
        
        # OOM 통계 출력
        if hasattr(pl_module, 'oom_count') and pl_module.oom_count > 0:
            logger.warning(f"[Epoch {trainer.current_epoch}] Total OOMs so far: {pl_module.oom_count}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        훈련 에포크 종료 시 실행되는 콜백 메서드
        
        에포크가 완료될 때 OOM 통계를 최종 확인하고 로깅합니다.
        
        Args:
            trainer (pl.Trainer): PyTorch Lightning 트레이너
            pl_module (VLMModule): 모델 모듈
            
        Processing:
            1. 모델 모듈의 OOM 카운트 확인
            2. OOM이 발생했다면 경고 메시지 출력
            
        Side Effects:
            - OOM 통계 경고 로그 (발생한 경우)
        """
        if hasattr(pl_module, 'oom_count') and pl_module.oom_count > 0:
            logger.warning(f"[Epoch {trainer.current_epoch}] Epoch ended with {pl_module.oom_count} total OOMs")

class LogSamplesCallback(pl.Callback):
    """
    검증 샘플 로깅 콜백
    
    검증 단계에서 모델의 출력 샘플을 생성하고 WandB에 로깅합니다.
    이미지, 입력 텍스트, 모델 예측을 시각적으로 추적할 수 있습니다.
    
    Args:
        tokenizer: HuggingFace 토크나이저 (텍스트 디코딩용)
        num_samples (int): 로깅할 샘플 수 (기본값: 16)
        max_new_tokens (int): 생성할 최대 토큰 수 (기본값: 256)
        
    Attributes:
        tok: 저장된 토크나이저
        n (int): 샘플 수
        m (int): 최대 생성 토큰 수
        last_logged_epoch (int): 마지막 로깅된 에포크 (중복 방지용)
        
    Methods:
        on_validation_epoch_end: 검증 완료 시 샘플 생성 및 로깅
        _denormalize_image: 이미지 정규화 해제 (시각화용)
    """
    def __init__(self, tokenizer, num_samples=16, max_new_tokens=256):
        """
        LogSamplesCallback 초기화
        
        Args:
            tokenizer: 텍스트 디코딩용 토크나이저
            num_samples (int): 로깅할 샘플 수
            max_new_tokens (int): 생성할 최대 토큰 수
        """
        self.tok, self.n, self.m = tokenizer, num_samples, max_new_tokens
        self.last_logged_epoch = -1
    
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        검증 에포크 종료 시 샘플 생성 및 로깅
        
        검증이 완료된 후 일부 샘플에 대해 모델 예측을 생성하고
        이미지, 입력, 예측을 WandB 테이블로 로깅합니다.
        
        Args:
            trainer (pl.Trainer): PyTorch Lightning 트레이너
            pl_module (VLMModule): 모델 모듈
            
        Processing:
            1. 중복 로깅 방지 체크 (last_logged_epoch)
            2. 모델을 evaluation 모드로 설정
            3. 검증 데이터에서 배치 샘플링
            4. 모델 생성 함수로 예측 수행
            5. 입력 텍스트 디코딩
            6. 이미지 정규화 해제 (시각화용)
            7. WandB 테이블 생성 및 업로드
            8. 모델을 다시 training 모드로 복원
            
        Returns:
            None
            
        Side Effects:
            - WandB에 "val_samples" 테이블 로깅
            - 모델 모드 변경 (eval → train)
            - 콘솔에 로깅 완료 메시지 출력
            
        Error Handling:
            - 모든 예외를 포착하여 로깅 실패 시에도 훈련 지속
            - finally 블록에서 모델 모드 복원 보장
        """
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
                    
                    # 이미지 정규화 해제 (SigLIP 기준)
                    img_denorm = self._denormalize_image(img)
                    
                    input_str = input_texts[i] if input_texts is not None else "<no input>"
                    # 입력 텍스트 정리 (불필요한 특수문자 제거)
                    if input_str != "<no input>":
                        input_str = self._clean_text_for_logging(input_str)
                    
                    pred_str = preds[i] if i < len(preds) else "<no prediction>"
                    # 예측 텍스트도 정리
                    if pred_str != "<no prediction>":
                        pred_str = self._clean_text_for_logging(pred_str)
                    
                    img_path = image_paths[i] if image_paths is not None else "<no path>"
                    
                    tbl.add_data(i, wandb.Image(img_denorm.cpu()), img_path, input_str, pred_str)
                
                trainer.logger.experiment.log({"val_samples": tbl}, commit=False)
                logger.info(f"Logged {min(actual_n, len(preds))} validation samples")
        
        except Exception as e:
            logger.error(f"Error in sample logging: {e}")
        finally:
            pl_module.train()
    
    def _clean_text_for_logging(self, text: str) -> str:
        """
        로깅용 텍스트 정리 함수
        
        WandB 로깅 시 불필요한 특수문자나 연속된 기호들을 정리합니다.
        
        Args:
            text (str): 정리할 텍스트
            
        Returns:
            str: 정리된 텍스트
        """
        import re
        
        # 연속된 느낌표 제거 (3개 이상)
        text = re.sub(r'!{3,}', '', text)
        
        # 줄 끝의 단독 느낌표 제거
        text = re.sub(r'\n!\s*$', '', text)
        text = re.sub(r'\n!\s*\n', '\n', text)
        
        # 연속된 물음표나 기타 특수문자도 정리
        text = re.sub(r'\?{3,}', '?', text)
        text = re.sub(r'\.{4,}', '...', text)
        
        # 기본 정리
        text = text.strip()
        
        return text

    def _denormalize_image(self, img_tensor):
        """
        이미지 정규화 해제 (WandB 시각화용)
        
        모델 입력용으로 정규화된 이미지 텐서를 원본에 가까운
        RGB 값으로 복원합니다. SigLIP/ImageNet 정규화를 역변환합니다.
        
        Args:
            img_tensor (torch.Tensor): 정규화된 이미지 텐서 (C, H, W)
                - C=3 (RGB channels)
                - H, W: 이미지 높이, 너비
                - 값 범위: 정규화된 범위 (평균 제거, 표준편차 나눔 적용된 상태)
                
        Returns:
            torch.Tensor: 정규화 해제된 이미지 텐서 (C, H, W)
                - 값 범위: [0, 1] (WandB 시각화에 적합)
                - 같은 디바이스 및 dtype 유지
                
        Processing:
            1. SigLIP/ImageNet 정규화 파라미터 로드
               - mean = [0.485, 0.456, 0.406] (R, G, B)
               - std = [0.229, 0.224, 0.225] (R, G, B)
            2. GPU 텐서인 경우 파라미터를 같은 디바이스로 이동
            3. 역정규화 공식 적용: x = (normalized_x * std) + mean
            4. [0, 1] 범위로 클리핑 (시각화 안정성)
            
        Note:
            SigLIP은 ImageNet과 동일한 정규화를 사용합니다.
            이 함수는 WandB 업로드용 시각화에만 사용되며 모델 추론에는 영향을 주지 않습니다.
        """
        # SigLIP/ImageNet 정규화 파라미터
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # GPU 텐서인 경우 CPU로 이동
        if img_tensor.is_cuda:
            mean = mean.to(img_tensor.device)
            std = std.to(img_tensor.device)
        
        # 정규화 해제: x = (normalized_x * std) + mean
        denormalized = img_tensor * std + mean
        
        # [0, 1] 범위로 클리핑 (원본 이미지가 이 범위였다고 가정)
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized

# =============================================================================
# 4. main
# =============================================================================

def run_stages(args, stages=None, prev_ckpt=None):
    """
    다중 단계 훈련 총괄 함수
    
    PanoLLaVA의 3단계 훈련 파이프라인을 관리하고 실행합니다.
    단일 단계 또는 연속된 다중 단계 훈련을 지원합니다.
    
    Args:
        args (argparse.Namespace): 파싱된 명령줄 인자들
            - 모든 훈련 설정 (모델, 데이터, 하이퍼파라미터 등)
        stages (None, str, list, tuple, optional): 실행할 훈련 단계
            - None: 단일 단계 (args.stage 사용)
            - str: 지정된 단계 하나만 실행
            - list/tuple: 여러 단계를 순차적으로 실행
        prev_ckpt (str, optional): 이전 체크포인트 경로
            
    Returns:
        str: 최종 단계의 최고 성능 체크포인트 경로
        
    Training Pipeline:
        1. vision: Vision encoder 학습 (VICReg loss)
        2. resampler: Resampler + Projection 학습 (AR loss)
        3. finetune: 전체 모델 또는 LoRA 학습 (AR loss)
        
    Processing:
        - 단일/다중 단계 분기 처리
        - 각 단계별 체크포인트 체인 관리
        - 총 훈련 시간 측정 및 로깅
        - 예외 발생 시 적절한 에러 처리
        
    Side Effects:
        - 콘솔에 진행 상황 로깅
        - 체크포인트 파일 생성
        - WandB 실험 로깅
        
    Error Handling:
        - 모든 예외를 포착하고 재발생
        - finally 블록에서 총 훈련 시간 로깅
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
    단일 훈련 단계 실행 핵심 함수
    
    하나의 훈련 단계를 완전히 실행하는 함수입니다.
    데이터 준비, 모델 초기화, 훈련 실행, 체크포인트 저장을 담당합니다.
    
    Args:
        args (argparse.Namespace): 훈련 설정 파라미터
        stage (str): 실행할 훈련 단계 ('vision', 'resampler', 'finetune')
        prev_ckpt (str, optional): 이전 단계의 체크포인트 경로
        
    Returns:
        str: 생성된 최고 성능 체크포인트의 경로
        
    Processing Flow:
        1. 단계별 기본값 적용 및 설정 백업
        2. VLMDataModule 초기화 (데이터로더 설정)
        3. 체크포인트 로딩 및 단계 변경 감지
        4. VLMModule 초기화 (모델 + Lightning 래퍼)
        5. 체크포인트에서 가중치 로딩
        6. WandB 로거 및 콜백 설정
        7. PyTorch Lightning Trainer 초기화
        8. 훈련 실행 (trainer.fit)
        9. LoRA 가중치 별도 저장 (필요시)
        10. 최종 모델 저장
        11. 원본 설정값 복원
        
    Configuration Management:
        - Config.STAGE_DEFAULTS에서 단계별 기본값 적용
        - 원본 설정값 백업 후 훈련 완료 시 복원
        - 단계 변경 시 적절한 체크포인트 처리
        
    Checkpoint Handling:
        - 이전 체크포인트에서 가중치 로딩
        - 단계 변경 감지 시 새로운 훈련으로 처리
        - 최고 성능 체크포인트 자동 저장
        
    Error Handling:
        - 데이터모듈 초기화 실패 처리
        - 모델 초기화 실패 처리
        - 훈련 실패 처리
        - 모델 저장 실패 처리
        
    Side Effects:
        - 체크포인트 디렉토리 생성
        - WandB 실험 초기화
        - 콘솔 로깅 출력
        - args 객체의 임시 수정 (완료 후 복원)
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
            image_size=args.image_size,
            tokenizer_name=args.lm_name,
            max_text_length=args.max_text_length,
            crop_strategy=args.crop_strategy,
            system_msg=args.system_msg,
            overlap_ratio=args.overlap_ratio,
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
        lit_model = VLMModule(
            vision_name=args.vision_name, 
            lm_name=args.lm_name, 
            resampler=args.resampler, 
            stage=stage, 
            lr=args.lr, 
            max_text_length=args.max_text_length,
            vicreg_loss_weight=args.vicreg_loss_weight,  # VICReg loss weight 추가
            overlap_ratio=args.overlap_ratio,
            # LoRA 파라미터들
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules
        )
        
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
    run_name = args.wandb_name if args.wandb_name else f"{stage}_{Path(args.csv_train).stem}_{int(time.time())}"
    wandb_dir = "./runs"
    Path(wandb_dir).mkdir(exist_ok=True)
    
    # WandB 설정 - 모든 인자 포함
    wandb_config = {
        "stage": stage,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "vicreg_loss_weight": getattr(args, "vicreg_loss_weight", None),
        "overlap_ratio": args.overlap_ratio,
        "vision_name": args.vision_name,
        "lm_name": args.lm_name,
        "resampler": args.resampler,
        "max_text_length": args.max_text_length,
        "csv_train": args.csv_train,
        "csv_val": args.csv_val,
        "num_workers": args.num_workers,
        "crop_strategy": args.crop_strategy,
        "image_size": args.image_size,
        "prefix": args.prefix,
        "system_msg": args.system_msg,
        # LoRA 관련 파라미터들
        "use_lora": getattr(args, "use_lora", False),
        "lora_rank": getattr(args, "lora_rank", None),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", None),
        "lora_target_modules": getattr(args, "lora_target_modules", None),
        "save_lora_only": getattr(args, "save_lora_only", False),
        # 추가 정보
        "resume_from": args.resume_from,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
    }
    
    # 콜백 설정
    callbacks = []
    
    # 배치 크기 모니터링 콜백 (항상 추가)
    callbacks.append(BatchSizeMonitorCallback())
    
    # 체크포인트 콜백
    ckpt_dir = f"./runs/{args.prefix}_{args.crop_strategy}_{stage}_{args.resampler}"
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
        patience=2,
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
        # 모델의 하이퍼파라미터도 WandB에 추가로 기록
        if wandb_logger.experiment:
            # VLMModule의 hparams를 WandB config에 추가
            model_hparams = lit_model.hparams
            for key, value in model_hparams.items():
                if key not in wandb_config:  # 중복 방지
                    wandb_logger.experiment.config[key] = value
            
            # 모델 내부 설정값들도 추가로 기록
            try:
                model_config = {
                    "model_vicreg_loss_weight": lit_model.model.vicreg_loss_weight,
                    "model_vicreg_overlap_ratio": lit_model.model.vicreg_overlap_ratio,
                    "model_max_text_length": lit_model.model.max_text_length,
                    "model_language_model_name": lit_model.model.language_model.config.name_or_path if hasattr(lit_model.model.language_model.config, 'name_or_path') else 'unknown',
                    "model_vision_encoder_name": getattr(lit_model.model.vision_encoder.config, 'name_or_path', 'unknown'),
                }
                
                # LoRA 정보 (활성화된 경우)
                if lit_model.use_lora:
                    lora_info = lit_model.model.get_lora_info()
                    model_config.update({f"model_{k}": v for k, v in lora_info.items()})
                
                wandb_logger.experiment.config.update(model_config)
                logger.info(f"Added {len(model_hparams)} model hyperparameters + {len(model_config)} model config to WandB")
            except Exception as config_error:
                logger.warning(f"Failed to add model config to WandB: {config_error}")
                logger.info(f"Added {len(model_hparams)} model hyperparameters to WandB config")
    except Exception as e:
        logger.warning(f"Failed to initialize WandB logger: {e}. Training will continue without WandB.")
        wandb_logger = None
    
    # 트레이너 초기화
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval = 0.5,
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
        
        # LoRA 사용시 추가 저장
        if stage == "finetune" and args.use_lora:
            try:
                # LoRA 가중치 항상 저장 (eval에서 사용하기 위해)
                lora_save_path = str(Path(ckpt_dir) / "lora_weights")
                lit_model.model.save_lora_weights(lora_save_path)
                logger.info(f"✓ LoRA weights saved to: {lora_save_path}")
                
                # LoRA 정보 출력
                lora_info = lit_model.model.get_lora_info()
                if lora_info.get("is_lora_enabled", False):
                    logger.info("✅ LoRA training completed successfully!")
                    logger.info(f"📊 LoRA configuration:")
                    logger.info(f"   - Rank: {lora_info.get('lora_r', 'N/A')}")
                    logger.info(f"   - Alpha: {lora_info.get('lora_alpha', 'N/A')}")
                    logger.info(f"   - Dropout: {lora_info.get('lora_dropout', 'N/A')}")
                    logger.info(f"   - Target modules: {lora_info.get('target_modules', 'N/A')}")
                
                # --save-lora-only 옵션이 활성화되면 전체 모델 체크포인트 저장 생략
                if args.save_lora_only:
                    logger.info("💾 save-lora-only enabled: Skipping full model checkpoint save")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to save LoRA weights: {e}")
        
    except Exception as e:
        logger.error(f"Training failed for stage {stage}: {e}")
        raise
    
    # 새로운 인터페이스와 호환되는 모델 저장 방식
    # --save-lora-only 옵션이 활성화되면 LoRA 사용 시 전체 모델 저장 생략
    skip_full_model_save = (stage == "finetune" and args.use_lora and args.save_lora_only)
    
    if not skip_full_model_save:
        try:
            # 새로운 인터페이스와 호환되는 형태로 저장
            logger.info("💾 새로운 인터페이스 호환 모델 저장 중...")
            
            # HuggingFace 스타일 저장 (새 인터페이스에서 바로 로딩 가능)
            hf_style_dir = Path(ckpt_dir) / "hf_model"
            lit_model.model.save_pretrained(str(hf_style_dir))
            logger.info(f"✅ HuggingFace 스타일 모델 저장: {hf_style_dir}")
            
            # 기존 방식도 유지 (호환성)
            final_model_path = str(Path(ckpt_dir) / "model_final.safetensors")
            save_checkpoint_safely(lit_model.state_dict(), final_model_path)
            logger.info(f"✅ 기존 방식 모델 저장: {final_model_path}")
            
            # 간편 로딩을 위한 심볼릭 링크 또는 복사 생성
            try:
                best_model_simplified = Path(ckpt_dir) / "panorama_model"
                if best_model_simplified.exists():
                    import shutil
                    shutil.rmtree(best_model_simplified)
                
                # HuggingFace 스타일 디렉토리를 간편한 이름으로 복사
                import shutil
                shutil.copytree(hf_style_dir, best_model_simplified)
                logger.info(f"✅ 간편 로딩용 모델 저장: {best_model_simplified}")
                logger.info(f"   사용법: model = PanoramaVLM.from_pretrained('{best_model_simplified}')")
                
            except Exception as link_e:
                logger.warning(f"⚠️ 간편 로딩용 모델 생성 실패: {link_e}")
            
        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")
    else:
        logger.info("💾 Skipping full model save (save-lora-only enabled)")
    
    # 훈련 완료 후 사용법 안내 출력
    logger.info("=" * 80)
    logger.info("🎉 훈련 완료! 모델 사용법:")
    logger.info("=" * 80)
    
    best_ckpt_path = ckpt_cb.best_model_path
    logger.info(f"📂 생성된 파일들:")
    logger.info(f"   - Lightning 체크포인트: {best_ckpt_path}")
    if Path(ckpt_dir + "/hf_model").exists():
        logger.info(f"   - HuggingFace 모델: {ckpt_dir}/hf_model")
    if Path(ckpt_dir + "/panorama_model").exists():
        logger.info(f"   - 간편 로딩용: {ckpt_dir}/panorama_model")
    if Path(ckpt_dir + "/lora_weights").exists():
        logger.info(f"   - LoRA 가중치: {ckpt_dir}/lora_weights")
    
    logger.info(f"")
    logger.info(f"🚀 새로운 간편 사용법:")
    logger.info(f"   # 방법 1: Lightning 체크포인트 (LoRA 자동 감지)")
    logger.info(f"   model = PanoramaVLM.from_checkpoint('{best_ckpt_path}')")
    logger.info(f"")
    
    if Path(ckpt_dir + "/panorama_model").exists():
        logger.info(f"   # 방법 2: HuggingFace 스타일 (가장 간편)")
        logger.info(f"   model = PanoramaVLM.from_pretrained('{ckpt_dir}/panorama_model')")
        logger.info(f"")
    
    if Path(ckpt_dir + "/hf_model").exists():
        logger.info(f"   # 방법 3: 디렉토리에서 자동 감지")
        logger.info(f"   model = PanoramaVLM.from_pretrained('{ckpt_dir}')")
        logger.info(f"")
    
    logger.info(f"💡 빠른 추론 테스트:")
    logger.info(f"   python simple_inference.py \\")
    logger.info(f"     --checkpoint '{best_ckpt_path}' \\")
    logger.info(f"     --image your_panorama.jpg")
    logger.info("=" * 80)
    
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
    p.add_argument("--prefix", default="panorama-vlm")
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
    p.add_argument("--overlap-ratio", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-text-length", type=int, default=32)
    p.add_argument("--system-msg", type=str, default=None,
                   help="커스텀 시스템 메시지 (기본값: 'You are a helpful assistant.')")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--wandb-project", default="panorama-vlm")
    p.add_argument("--wandb-name",    default=None)
    
    # LoRA 관련 파라미터들
    p.add_argument("--use-lora", action="store_true", 
                   help="finetune 단계에서 LoRA 사용")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank (기본값: 16)")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA alpha parameter (기본값: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.1,
                   help="LoRA dropout rate (기본값: 0.1)")
    p.add_argument("--lora-target-modules", nargs="*", default=None,
                   help="LoRA를 적용할 모듈들 (기본값: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)")
    p.add_argument("--save-lora-only", action="store_true",
                   help="LoRA 가중치만 저장 (전체 모델 대신)")
    
    args = p.parse_args()

    # 단일/전체 스테이지 학습 통합
    if args.stages is not None and len(args.stages) > 0:
        stages = args.stages if isinstance(args.stages, list) else args.stages.split()
        prev_ckpt = args.resume_from if args.resume_from else None
        run_stages(args, stages, prev_ckpt=prev_ckpt)
    else:
        run_stages(args)
