"""Language model wrapper for Panorama VLM."""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)

class LanguageModel(nn.Module):
    def __init__(self, model_name: str, use_cache: bool = False):
        super().__init__()
        self.model_name = model_name
        
        # Load kwargs
        load_kwargs = {
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }
        
        # DeepSeek-R1 / QwQ specific handling
        lm_name_lower = model_name.lower()
        if any(keyword in lm_name_lower for keyword in ['deepseek-r1', 'qwq', 'r1-distill']):
            load_kwargs["enable_think"] = False
            logger.info("Disabling 'think' for reasoning model")

        logger.info(f"Loading generic LLM: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        if hasattr(self.model, "config"):
            self.model.config.use_cache = use_cache
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._setup_tokenizer()
        
    def _setup_tokenizer(self):
        """Ensures tokenizer has pad, eos, and special vision tokens."""
        tokens_added = False
        
        # 1. Ensure pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token = eos_token: '{self.tokenizer.eos_token}'")
            else:
                self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'})
                tokens_added = True
                logger.info("Added eos_token and pad_token: '<|endoftext|>'")

        # 2. Add vision tokens
        special_tokens_to_add = []
        vision_token = '<|vision|>'
        if not any(vision_token in str(token) for token in self.tokenizer.additional_special_tokens):
            special_tokens_to_add.append(vision_token)
        
        # 3. Ensure eos_token is present if needed
        if self.tokenizer.eos_token != '<|endoftext|>' and self.tokenizer.eos_token is not None:
             # Basic check, might skip complex logic to avoid bloat
             pass

        if special_tokens_to_add:
            added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
            if added > 0:
                tokens_added = True
                logger.info(f"Added special tokens: {special_tokens_to_add}")

        if tokens_added:
            self.resize_token_embeddings()

        self.tokenizer.padding_side = "right" # Default for training

    def resize_token_embeddings(self):
        old_embeddings = self.model.get_input_embeddings().weight.size(0)
        self.model.resize_token_embeddings(len(self.tokenizer))
        new_embeddings = self.model.get_input_embeddings().weight.size(0)
        logger.info(f"Resized embeddings: {old_embeddings} -> {new_embeddings}")

    def setup_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.1, target_modules: Optional[List[str]] = None) -> bool:
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. LoRA skipped.")
            return False
            
        if target_modules is None:
            # Default to all linear layers commonly used
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        try:
            config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()
            return True
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            return False

    def load_lora_weights(self, path: str):
        if not PEFT_AVAILABLE:
            return False
        
        logger.info(f"Loading LoRA weights from {path}")
        try:
            # If already PEFT model, load adapter
            if hasattr(self.model, "load_adapter"):
                self.model.load_adapter(path, adapter_name="default")
                self.model.set_adapter("default")
            else:
                # Load as PeftModel
                self.model = PeftModel.from_pretrained(self.model, path)
            return True
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            return False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    @property
    def config(self):
        return self.model.config
