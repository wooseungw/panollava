"""PyTorch Lightning Module for CORA Training."""

import logging
import os
import torch
import lightning as pl
from typing import Dict, Any, Optional, List
from cora.config.schema import CORAConfig, ModelConfig, StageConfig
from cora.models.vlm import PanoramaVLM
from cora.training.losses import VICRegLoss
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class PanoramaTrainingModule(pl.LightningModule):
    """
    Lightning Module for training PanoramaVLM.
    Handles 3-stage training:
    1. Vision (VICReg): Vision Encoder (optional) + Resampler + Projector
    2. Resampler (VICReg): Vision (optional) + Resampler + Projector
    3. Finetune (CausalLM): Vision (optional) + Resampler + Projector + LLM (LoRA)
    """

    def __init__(
        self,
        config: CORAConfig,
        stage: str = "finetune",
        vision_trainable_blocks: int = 0
    ):
        super().__init__()
        self.save_hyperparameters() # Saves all init args
        self.config = config
        self.model_config = config.models
        self.stage = stage
        self.vision_trainable_blocks = vision_trainable_blocks
        
        # Resolve Stage Config
        self.stage_config = self.config.training.stage_configs.get(stage, StageConfig())
        
        # 1. Initialize Model
        # We assume PanoramaVLM is the main entry point
        self.model = PanoramaVLM(self.config)
        
        # 2. Setup LoRA if needed (Stage 3)
        self.use_lora = False
        # 2. Setup LoRA check (handled in VLM init)
        self.use_lora = False
        if stage == "finetune" and self.config.lora and self.config.lora.use_lora:
            self.use_lora = True
        
        # 3. Setup Freezing/Unfreezing based on Stage
        self._setup_freezing()
        
        # 4. Losses
        # VICReg for stages 1 & 2
        # CausalLM for stage 3 is handled inside PanoramaVLM forward
        if stage in ("vision", "resampler"):
            # Use weights from StageConfig
            # Note: vicreg_loss_weight is a global multiplier often used. 
            # Individual terms are similarity/variance/covariance weights.
            # If vicreg_loss_weight is > 0, we can use it to scale defaults or direct usage.
            
            # Legacy logic used a global weight to scale the terms.
            scale = self.stage_config.vicreg_loss_weight if self.stage_config.vicreg_loss_weight > 0 else 1.0
            
            self.vicreg_loss = VICRegLoss(
                similarity_weight=self.stage_config.vicreg_similarity_weight * scale,
                variance_weight=self.stage_config.vicreg_variance_weight * scale,
                covariance_weight=self.stage_config.vicreg_covariance_weight * scale,
                overlap_ratio=self.config.image_processing.overlap_ratio,
                vicreg_mode=self.stage_config.vicreg_mode
            )

    def _setup_freezing(self):
        """Freeze/Unfreeze parameters based on stage."""
        # 1. Freeze everything first
        self.model.requires_grad_(False)
        
        # Vision Encoder Unfreezing Logic (Shared)
        def unfreeze_vision(blocks):
             if blocks != 0:
                 # Logic to unfreeze last N blocks of vision encoder
                 # This depends on specific backbone structure.
                 if hasattr(self.model.vision_encoder, "unfreeze_last_n_layers"):
                     self.model.vision_encoder.unfreeze_last_n_layers(blocks)
                 else:
                     logger.warning("Vision encoder does not support block-wise unfreezing yet.")

        if self.stage == "vision":
            # Stage 1: Vision (VICReg)
            # Train: Resampler, VICReg Projector
            # Optional: Vision Encoder
            unfreeze_vision(self.vision_trainable_blocks)
            
            if hasattr(self.model, "resampler"):
                self.model.resampler.requires_grad_(True)
            
            # Use specific VICReg projector if available, or shared projector
            if hasattr(self.model, "vicreg_projector") and self.model.vicreg_projector:
                self.model.vicreg_projector.requires_grad_(True)
            elif hasattr(self.model, "projector"):
                self.model.projector.requires_grad_(True)

        elif self.stage == "resampler":
            # Stage 2: Resampler (VICReg)
            unfreeze_vision(self.vision_trainable_blocks)
            self.model.resampler.requires_grad_(True)
            
            if hasattr(self.model, "vicreg_projector") and self.model.vicreg_projector:
                self.model.vicreg_projector.requires_grad_(True)
            elif hasattr(self.model, "projector"):
                self.model.projector.requires_grad_(True)

        elif self.stage == "finetune":
            # Stage 3: Finetune (CausalLM)
            unfreeze_vision(self.vision_trainable_blocks)
            self.model.resampler.requires_grad_(True)
            if hasattr(self.model, "projector"):
                self.model.projector.requires_grad_(True)
            
            if self.use_lora:
                self._unfreeze_lora()
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Stage '{self.stage}': Trainable params: {trainable:,}/{total:,} ({trainable/total:.1%})")

    def _unfreeze_lora(self):
        """Unfreeze LoRA parameters."""
        count = 0
        for name, p in self.model.named_parameters():
             if "lora" in name:
                 p.requires_grad_(True)
                 count += 1
        logger.info(f"Unfrozen {count} LoRA parameters")

    def forward(self, **kwargs):
        return self.model(stage=self.stage, **kwargs)

    def training_step(self, batch, batch_idx):
        # 1. Forward
        # Batch should contain pixel_values, input_ids, etc.
        # Model forward handles logic based on stage
        outputs = self(**batch)
        loss = outputs.get("loss", 0.0)
        
        # 2. Add VICReg Loss if in early stages
        if self.stage in ("vision", "resampler"):
            # Outputs might contain features or model computed loss
            # If model returns loss=0 for vision/resampler, it means we must compute VICReg here.
            # Check what forward returns for vision stage:
            # { "vicreg_features": ..., "batch_size": ..., "num_views": ... }
            
            if "vicreg_features" in outputs:
                feats = outputs["vicreg_features"]
                bs = outputs["batch_size"]
                nv = outputs["num_views"]
                
                vicreg_l = self.vicreg_loss(feats, bs, nv)
                loss = vicreg_l # Overlay loss (since model loss is likely 0 or irrelevant)
                
                self.log("train_vicreg_loss", vicreg_l, prog_bar=True, sync_dist=True)
            
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=self.device)

        anchor_param = next((p for p in self.model.parameters() if p.requires_grad), None)
        if anchor_param is not None and (not loss.requires_grad or loss.grad_fn is None):
            loss = loss + (anchor_param.reshape(-1)[0] * 0.0)

        # Log
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.get("loss", 0.0)
        
        if self.stage in ("vision", "resampler") and "vicreg_features" in outputs:
                feats = outputs["vicreg_features"]
                bs = outputs["batch_size"]
                nv = outputs["num_views"]
                loss = self.vicreg_loss(feats, bs, nv)
                self.log("val_vicreg_loss", loss, prog_bar=True, sync_dist=True)
                
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Differential Learning Rates
        if isinstance(self.stage_config, dict):
            lr = self.stage_config.get("lr", self.config.training.learning_rate)
        else:
            lr = getattr(self.stage_config, "lr", self.config.training.learning_rate)
        
        param_groups = []
        
        # Group parameters
        vision_params = []
        resampler_params = []
        projector_params = []
        llm_params = []
        other_params = []
        
        for name, p in self.model.named_parameters():
             if not p.requires_grad: continue
             
             if "vision_encoder" in name: vision_params.append(p)
             elif "resampler" in name: resampler_params.append(p)
             elif "projector" in name: projector_params.append(p)
             elif "language_model" in name: llm_params.append(p)
             else: other_params.append(p)
             
        # Add groups
        if vision_params:
            param_groups.append({"params": vision_params, "lr": lr * 0.1, "weight_decay": 0.01})
        if resampler_params:
            param_groups.append({"params": resampler_params, "lr": lr, "weight_decay": 0.05})
        if projector_params:
            param_groups.append({"params": projector_params, "lr": lr, "weight_decay": 0.05})
        if llm_params:
            param_groups.append({"params": llm_params, "lr": lr, "weight_decay": 0.01}) # Usually LoRA
        if other_params:
             param_groups.append({"params": other_params, "lr": lr})
             
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)
        
        # Scheduler
        if self.trainer.max_epochs > 0:
             steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) if self.trainer.datamodule else 100
             total_steps = steps_per_epoch * self.trainer.max_epochs
             warmup_steps = int(0.1 * total_steps)
             scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
             return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        return optimizer
