"""Main Panorama VLM Model Wrapper."""

from __future__ import annotations

import logging
from typing import Optional, List, Union, Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from cora.config.schema import CORAConfig
from cora.models.vision_encoder import VisionBackbone
from cora.models.language_model import LanguageModel
from cora.models.resampler import ResamplerModule
from cora.models.projectors import PanoramaProjector, VICRegProjector
from cora.models.language_fusion import LanguageFusion
from cora.utils import resolve_module_dtype

logger = logging.getLogger(__name__)

class PanoramaVLM(nn.Module):
    """
    Main VLM orchestrator class.
    Combines Vision Encoder, Resampler, Projectors, and Language Model.
    Supports 3-stage training:
    1. Vision Stage: Vision + Resampler + VICReg
    2. Resampler Stage: Vision + Resampler + Projector + Frozen LLM
    3. Finetune Stage: Full finetuning (with LoRA usually)
    """
    def __init__(self, config: CORAConfig):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder
        logger.info("Initializing Vision Encoder...")
        self.vision_encoder = VisionBackbone(
            vision_name=config.models.vision_name,
            use_vicreg_norm=config.models.use_vicreg_norm,
            backbone_type=config.models.vision_backbone_type,
            # Pass any extra kwargs if needed, e.g. from config.models.vision_kwargs
        )
        self.vision_hidden_size = self.vision_encoder.hidden_size

        # 2. Resampler
        logger.info("Initializing Resampler...")
        self.resampler_module = ResamplerModule(config.models, self.vision_hidden_size)
        self.resampler = self.resampler_module # Wrapper exposes .forward
        self.latent_dimension = self.resampler_module.output_dim

        # 3. VICReg Projector (Stage 1)
        if config.models.use_vicreg_projector:
            self.vicreg_projector = VICRegProjector(
                in_dim=self.latent_dimension,
                out_dim=config.models.vicreg_projector_dim or self.latent_dimension,
                depth=config.models.vicreg_projector_depth,
                use_ln=config.models.vicreg_projector_ln,
            )
        else:
            self.vicreg_projector = nn.Identity()

        # 4. Language Model
        logger.info("Initializing Language Model...")
        self.language_model_wrapper = LanguageModel(config.models.language_model_name)
        self.language_model = self.language_model_wrapper.model
        self.tokenizer = self.language_model_wrapper.tokenizer
        self.llm_hidden_size = self.language_model.config.hidden_size
        
        # 5. Panorama Projector (Vision -> LLM)
        logger.info("Initializing Panorama Projector...")
        self.projector = PanoramaProjector(
            config=config.models,
            latent_dimension=self.latent_dimension,
            language_hidden_size=self.llm_hidden_size,
        )
        self.vision_to_language_projection = self.projector # Alias

        # 6. Text Projector (Optional, for Stage 2/3 if needed)
        if config.models.use_text_projection:
            self.text_projection = nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        else:
            self.text_projection = nn.Identity()
            
        # 7. Language Fusion Utility
        vision_token_str = "<|vision|>"
        vision_token_id = self.tokenizer.convert_tokens_to_ids(vision_token_str)
        if vision_token_id == self.tokenizer.unk_token_id:
             # Fallback if not added (LanguageModel wrapper should have added it)
             logger.warning(f"Vision token {vision_token_str} not found, using unk_token_id")
             
        # Get max_text_length safely from finetune config (which might be StageConfig object or dict)
        finetune_cfg = config.training.stage_configs.get("finetune")
        max_text_len = 2048
        if finetune_cfg:
            max_text_len = getattr(finetune_cfg, "max_text_length", 2048) if not isinstance(finetune_cfg, dict) else finetune_cfg.get("max_text_length", 2048)

        self.fusion = LanguageFusion(
            language_model=self.language_model,
            tokenizer=self.tokenizer,
            vision_token_id=vision_token_id,
            ignore_index=-100,
            max_text_length=max_text_len,
        )
        
        # Apply LoRA if configured
        if config.lora.use_lora:
            logger.info("Applying LoRA...")
            self.language_model_wrapper.setup_lora(
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                target_modules=config.lora.target_modules
            )

        self.dtype_cache: Dict[str, torch.dtype] = {}

    def get_vision_tower(self):
        return self.vision_encoder

    def _process_resampler(
        self,
        pixel_values: torch.Tensor,
        batch_size: int,
        num_views: int
    ) -> torch.Tensor:
        """
        Runs vision encoder -> resampler.
        Returns: [BV, S, LatentDim]
        """
        # Vision Encoder
        # pixel_values: [B, V, C, H, W] or [B, C, H, W]
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs["vision_features"] # [BV, S, VisionDim]
        
        # Resampler
        resampled_features = self.resampler_module(vision_features) # [BV, S, LatentDim]
        
        return resampled_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "finetune",
        **kwargs
    ):
        """
        Forward pass handling different stages.
        """
        B_pixel = pixel_values.shape[0] if pixel_values is not None else 1
        num_views = pixel_values.shape[1] if pixel_values is not None and pixel_values.ndim == 5 else 1

        if stage == "vision":
            # Stage 1: Vision Self-Supervised (VICReg)
            # Only needs pixel_values
            resampled_features = self._process_resampler(pixel_values, B_pixel, num_views)
            
            # Project for VICReg
            vicreg_feats = self.vicreg_projector(resampled_features) # [BV, S, OutDim]
            
            return {
                "vicreg_features": vicreg_feats,
                "batch_size": B_pixel,
                "num_views": num_views
            }

        elif stage in ("resampler", "finetune"):
            # Stage 2/3: Vision -> LLM
            
            # 1. Vision Path
            resampled_features = self._process_resampler(pixel_values, B_pixel, num_views)
            
            # 2. Project to LLM Dim
            vision_tokens = self.projector(
                resampled_features=resampled_features,
                batch_size=B_pixel,
                num_views=num_views,
                dtype_cache=self.dtype_cache,
                language_model=self.language_model
            ) # [B, TotalS, LLM_Dim] (stitched)
            
            # 3. Fuse with Text
            fused_inputs = self.fusion.fuse(
                vision_tokens=vision_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 4. LLM Forward
            outputs = self.language_model(
                inputs_embeds=fused_inputs["inputs_embeds"],
                attention_mask=fused_inputs["attention_mask"],
                labels=fused_inputs.get("labels"),
                return_dict=True
            )
            
            return outputs
            
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        B_pixel = pixel_values.shape[0] if pixel_values is not None else 1
        num_views = pixel_values.shape[1] if pixel_values is not None and pixel_values.ndim == 5 else 1

        # 1. Vision Path
        resampled_features = self._process_resampler(pixel_values, B_pixel, num_views)
        
        # 2. Project
        vision_tokens = self.projector(
            resampled_features=resampled_features,
            batch_size=B_pixel,
            num_views=num_views,
            dtype_cache=self.dtype_cache,
            language_model=self.language_model
        )
        
        # 3. Input Preparation
        # If input_ids not provided, create default prompt
        if input_ids is None:
            input_ids, attention_mask = self.fusion.create_default_prompt(B_pixel, pixel_values.device)
            
        # 4. Fuse
        # Use fuse() but distinct behavior for generation (usually we just want inputs_embeds)
        # However fuse() handles inserting vision tokens into placeholders.
        fused_inputs = self.fusion.fuse(
            vision_tokens=vision_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 5. Generate
        gen_kwargs = generation_config or {}
        # Merge typical defaults if not present
        if "max_new_tokens" not in gen_kwargs: gen_kwargs["max_new_tokens"] = 128
        
        outputs = self.language_model.generate(
            inputs_embeds=fused_inputs["inputs_embeds"],
            attention_mask=fused_inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )
        
        return outputs

