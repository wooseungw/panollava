"""Main Training Orchestrator for CORA."""

import logging
import os
from pathlib import Path
from typing import Optional, List
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from cora.config.schema import CORAConfig
from cora.config.manager import ConfigManager
from cora.training.module import PanoramaTrainingModule
from cora.training.callbacks import MetadataCallback
from cora.data.dataset import PanoramaDataModule
from cora.processors.processor import PanoramaProcessor
from cora.processors.images import PanoramaImageProcessor
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class CORATrainer:
    """
    Orchestrates the 3-stage training process for CORA.
    """
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config: CORAConfig = self.config_manager.config
        self.config_path = config_path
        
        # Setup Output Dir
        self.output_dir = Path(self.config.training.output_dir) / self.config.experiment["name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Processor
        self._setup_processor()
        
    def _setup_processor(self):
        """Initialize Processor (Image + Text)."""
        logger.info("Initializing Processors...")
        img_cfg = self.config.image_processing
        
        # Image Processor
        image_size = img_cfg.image_size
        if image_size is None:
            # Auto-infer from vision model name
            vname = self.config.models.vision_name.lower()
            if "patch16-256" in vname or "-256" in vname:
                image_size = [256, 256]
            elif "patch14-560" in vname or "-560" in vname:
                image_size = [560, 560]
            elif "patch16-384" in vname or "-384" in vname:
                image_size = [384, 384]
            elif "patch16-224" in vname or "-224" in vname:
                image_size = [224, 224]
            else:
                logger.warning(f"Could not infer image size from {vname}, defaulting to [224, 224]")
                image_size = [224, 224]
            logger.info(f"Auto-inferred image size: {image_size}")

        img_proc = PanoramaImageProcessor(
            image_size=image_size,
            crop_strategy=img_cfg.crop_strategy,
            fov_deg=img_cfg.fov_deg,
            overlap_ratio=img_cfg.overlap_ratio, # Use config overlap default
            normalize=True
        )
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.models.language_model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.processor = PanoramaProcessor(img_proc, tokenizer)
        
    def train(self):
        """Execute training stages."""
        stages = self.config.training.stages
        logger.info(f"Starting training pipeline with stages: {stages}")
        
        current_ckpt = self.config.training.resume_from_checkpoint
        
        for stage in stages:
            logger.info(f"=== Starting Stage: {stage} ===")
            current_ckpt = self._run_stage(stage, resume_checkpoint=current_ckpt)
            logger.info(f"=== Finished Stage: {stage} ===\n")
            
    def _run_stage(self, stage: str, resume_checkpoint: Optional[str] = None) -> str:
        """Run a single training stage."""
        stage_cfg = self.config.training.stage_configs.get(stage)

        def _stage_value(name: str, default):
            if stage_cfg is None:
                return default
            if isinstance(stage_cfg, dict):
                return stage_cfg.get(name, default)
            return getattr(stage_cfg, name, default)

        stage_batch_size = _stage_value("batch_size", self.config.training.batch_size)
        stage_accumulate = _stage_value("accumulate_grad_batches", self.config.training.accumulate_grad_batches)
        stage_vision_blocks = _stage_value("vision_trainable_blocks", self.config.training.vision_trainable_blocks)
        stage_epochs = _stage_value("epochs", self.config.training.max_epochs)
        
        # 1. DataModule
        datamodule = PanoramaDataModule(
            train_csv=self.config.data["train_csv"],
            val_csv=self.config.data["val_csv"],
            image_root=self.config.data.get("image_root"),
            processor=self.processor,
            batch_size=stage_batch_size,
            num_workers=self.config.training.num_workers
        )
        
        # 2. Lightning Module
        # Logic to carry over weights:
        # If resume_checkpoint is provided, PL handles loading.
        # But if we are switching stages, we might want to load weights but reset optimizer/step?
        # Usually, passing checkpoint to Trainer.fit() resumes everything.
        # If transitioning stages (e.g. Vision -> Resampler), we usually treat it as a new run 
        # but initialized from previous weights.
        
        module = PanoramaTrainingModule(
            config=self.config,
            stage=stage,
            vision_trainable_blocks=stage_vision_blocks
        )
        
        # If we have a checkpoint from previous stage, we ideally load weights manually
        # instead of full resume, to reset optimizers for the new stage.
        if resume_checkpoint and os.path.exists(resume_checkpoint):
             logger.info(f"Loading weights from previous stage: {resume_checkpoint}")
             # We use strict=False because stages might have different active params 
             # (though structural params are same, just required_grad differs).
             # Actually structure is same.
             checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
             state_dict = checkpoint["state_dict"]
             module.load_state_dict(state_dict, strict=False)
        
        # 3. Callbacks
        stage_dir = self.output_dir / stage
        ckpt_callback = ModelCheckpoint(
            dirpath=stage_dir,
            filename=f"{stage}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True
        )
        
        meta_callback = MetadataCallback(
            ckpt_dir=str(stage_dir),
            metadata={"stage": stage, "experiment": self.config.experiment["name"]},
            config_path=self.config_path
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        callbacks = [ckpt_callback, meta_callback, lr_monitor]
        
        # 4. Logger
        pl_logger = None
        if self.config.training.trackers:
             try:
                 pl_logger = WandbLogger(
                     project="cora-vlm",
                     name=f"{self.config.experiment['name']}-{stage}",
                     config=self.config.model_dump()
                 )
             except Exception:
                 logger.warning("Wandb not available/configured properly.")
                 
        # 5. Trainer
        trainer = pl.Trainer(
            max_epochs=stage_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.config.training.devices,
            precision=self.config.training.precision,
            callbacks=callbacks,
            logger=pl_logger,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=stage_accumulate,
            val_check_interval=self.config.training.val_check_interval,
            strategy=self.config.training.strategy # 'ddp' etc.
        )
        
        # 6. Fit
        trainer.fit(module, datamodule=datamodule)
        
        # Return best checkpoint path
        best_path = ckpt_callback.best_model_path
        logger.info(f"Best checkpoint for stage {stage}: {best_path}")
        
        return best_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    trainer = CORATrainer(args.config)
    trainer.train()
