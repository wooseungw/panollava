"""
Main Processor Class for CORA.
Unifies image and text processing directly using PanoramaImageProcessor and UniversalTextFormatter.
"""

from typing import Union, List, Optional, Dict, Any
from PIL import Image
from transformers import BatchEncoding, AutoTokenizer

from cora.processors.images import PanoramaImageProcessor
from cora.processors.text import UniversalTextFormatter

class PanoramaProcessor:
    """
    Multimodal processor for CORA:
    - Handles image preprocessing (AnyRes, etc.) via PanoramaImageProcessor.
    - Handles text formatting (Chat templates) via UniversalTextFormatter.
    - Handles final tokenization.
    """
    def __init__(self, image_processor: PanoramaImageProcessor, tokenizer: Any, system_prompt: Optional[str] = None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.text_formatter = UniversalTextFormatter(tokenizer, system_msg=system_prompt)
        
    def __call__(
        self,
        text: Union[str, List[str], List[Dict[str, str]]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: Optional[str] = "pt",
    ) -> BatchEncoding:
        """
        Process text and images.
        
        Args:
            text: Input text. Can be:
                  - A single string (raw prompt)
                  - A list of strings (batch of prompts)
                  - A list of chat messages (dictionaries with role/content)
            images: Input images. Can be single PIL Image or list of PIL Images.
            padding: Padding strategy.
            truncation: Truncation strategy.
            max_length: Maximum sequence length.
            return_tensors: Tensor return type (e.g. 'pt').
            
        Returns:
            BatchEncoding with 'input_ids', 'attention_mask', 'pixel_values' (if images provided).
        """
        
        # 1. Process Images
        pixel_values = None
        if images is not None:
            # Handles both single image and list of images internally
            # Returns [B, V, C, H, W] tensor (or similar structure depending on config)
            pixel_values_5d = self.image_processor(images) 
            
            # Flatten V dimension for some model architectures if needed, 
            # but usually model expects [B, V, C, H, W] or [BV, C, H, W].
            # Our model expects [B, V, C, H, W] for the VLM forward wrapper 
            # or [BV, C, H, W] if we flatten immediately.
            # Let's keep it as [B, V, C, H, W] for now to preserve structure.
            pixel_values = pixel_values_5d

        # 2. Process Text
        # Normalize input to list of strings
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            # Formatted chat messages -> string
            # Note: This assumes single conversation. For batch of conversations, we need list[list[dict]].
            # For simplicity, if passed a single conversation list, treat as single sample.
            text = [self.text_formatter.format_conversation(
                user_msg=msg['content'] if msg['role']=='user' else '',
                assistant_msg=msg['content'] if msg['role']=='assistant' else None
            ) for msg in text if msg['role'] == 'user'] # Simple heuristic, needs robustness for full chat
            
            # Better approach: If list of dicts, it's one conversation. 
            # If list of strings, it's a batch of raw prompts.
            # TODO: Robust batch chat handling.
            pass

        # Apply formatting if it's raw text (not yet formatted strings)
        # Here we assume the user might pass raw instructions.
        # Ideally, use apply_chat_template if available.
        # For this implementation, we assume `text` is already what needs to be tokenized 
        # OR we rely on the tokenizer to handle basic string list.
        
        # Let's use the UniversalTextFormatter to tokenize directly if generation logic is needed,
        # otherwise use standard tokenizer.
        
        tokenized = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
        
        data = dict(tokenized)
        if pixel_values is not None:
            data["pixel_values"] = pixel_values
            
        return BatchEncoding(data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
