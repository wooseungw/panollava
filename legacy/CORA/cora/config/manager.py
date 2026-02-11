import yaml
import json
import os
import re
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

from .schema import CORAConfig

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages loading, validation, and processing of CORA configurations."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config = self.load(config_path)

    @staticmethod
    def load(config_path: Union[str, Path]) -> CORAConfig:
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Validated CORAConfig object.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        print(f"[ConfigManager] Loading config from: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                raw_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                raw_config = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use .yaml or .json")

        # Create config object (validation happens here)
        try:
            config = CORAConfig(**raw_config)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise e

        # Automatic experiment naming
        if config.experiment.get("name") == "auto":
            config.experiment["name"] = _compute_experiment_name(config)
            print(f"[ConfigManager] Auto-generated experiment name: {config.experiment['name']}")

        return config

def _short_id_from_model_path(model_name: Optional[str], max_len: int = 12, *, siglip_include_patch_res: bool = False) -> str:
    """
    Extracts a short identifier from a HuggingFace model path.
    e.g., "google/siglip-so400m-patch14-384" -> "siglip-so400m"
    """
    if not model_name:
        return "unknown"
        
    name = model_name.strip("/").split("/")[-1]
    
    # Common prefix removal
    for prefix in ["param_", "lora_", "checkpoint-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Special handling for SigLIP/ViT
    if "siglip" in name.lower() or "vit" in name.lower():
        # logic to keep patch/res info if requested, otherwise shorten
        parts = name.split("-")
        keep = []
        for p in parts:
            if p in ["google", "model", "base", "large", "patch14", "patch16", "224", "384", "560"]:
                if "patch" in p or p.isdigit():
                    if not siglip_include_patch_res:
                        continue
            keep.append(p)
        name = "-".join(keep)
    
    # Qwen/LLaVA specific cleanup
    name = name.replace("Qwen2.5", "Qwen2.5").replace("Instruct", "").strip("-")
    
    # Generalized shortening
    # Remove version numbers/tags loosely if too long
    if len(name) > max_len:
         # Try to pick first few distinct segments
         parts = re.split(r'[-_]', name)
         short = "-".join(parts[:2])
         if len(short) > max_len:
             return short[:max_len]
         return short

    return name

def _compute_experiment_name(config: CORAConfig) -> str:
    """
    Generates a descriptive experiment name:
    {vision_model}_{lm_model}_{resampler}_{crop_strategy}_{extra}
    """
    
    # 1. Vision Model ID
    v_name = config.models.vision_name
    v_id = _short_id_from_model_path(v_name, max_len=15)
    
    # 2. LM Model ID
    lm_name = config.models.language_model_name
    lm_id = _short_id_from_model_path(lm_name, max_len=10)
    
    # 3. Resampler
    res_type = config.models.resampler_type
    
    # 4. Crop Strategy
    crop = config.image_processing.crop_strategy
    
    parts = [v_id, lm_id, res_type, crop]
    
    # 5. Extras (PE, LoRA)
    if config.models.use_projection_positional_encoding:
        parts.append("PE")
    if config.lora.use_lora:
        parts.append("LoRA")
        
    return "_".join(parts)
