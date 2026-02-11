"""Inference Generator for CORA."""

import logging
import torch
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from PIL import Image

from cora.config.manager import ConfigManager
from cora.training.module import PanoramaTrainingModule
from cora.processors.processor import PanoramaProcessor
from cora.processors.images import PanoramaImageProcessor
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class PanoramaGenerator:
    """
    Handles text generation inference for CORA models.
    Loads model from checkpoint and provides generate methods.
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # 1. Load Config
        # Try to find config.yaml in the same directory or parent
        config_path = self.checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
             config_path = self.checkpoint_path.parent.parent / "config.yaml"
             
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found near checkpoint: {self.checkpoint_path}")
            
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # 2. Load Model
        logger.info(f"Loading model from {checkpoint_path}...")
        self.module = PanoramaTrainingModule.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            config=self.config,
            stage="finetune", # Assume inference uses full model capabilities
            map_location="cpu"
        )
        self.model = self.module.model
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        # 3. Setup Processor
        self._setup_processor()
        
    def _setup_processor(self):
        """Initialize Processor."""
        img_cfg = self.config.image_processing
        
        img_proc = PanoramaImageProcessor(
            image_size=img_cfg.image_size,
            crop_strategy=img_cfg.crop_strategy,
            fov_deg=img_cfg.fov_deg,
            overlap_ratio=img_cfg.overlap_ratio,
            normalize=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.models.language_model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.processor = PanoramaProcessor(img_proc, tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Describe the panorama image.",
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text for a single image and prompt.
        """
        # Load Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        # Process inputs
        inputs = self.processor(
            text=[{"role": "user", "content": prompt}],
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            
        # Generate
        outputs = self.model.generate(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            generation_config=generation_config
        )
        
        # Decode
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Clean up prompt from output (depends on model behavior, 
        # but usually generate returns full sequence or just new tokens depending on implementation.
        # PanoramaVLM.generate calls language_model.generate which usually returns full sequence)
        
        # Simple cleanup if prompt is included (heuristic)
        # Ideally we should use the tokenizer to split, but assuming skip_special_tokens handles standard chat format...
        # Actually Qwen/Llama might include the input.
        # Let's rely on standard decoding for now.
        
        return generated_text

    @torch.inference_mode()
    def batch_generate(
        self,
        images: List[Union[str, Image.Image]],
        prompts: List[str],
        batch_size: int = 4,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Batch generation not fully implemented yet due to variable length padding complexity 
        in simple inference script. Recommended to use loop for now or Dataset class.
        """
        results = []
        for img, p in zip(images, prompts):
            results.append(self.generate(img, p, generation_config))
        return results
