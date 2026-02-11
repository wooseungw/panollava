"""Vision processor wrapper for HuggingFace vision models."""

from typing import Optional, Union
from PIL import Image
import torch


class VisionProcessorWrapper:
    """
    Wrapper for HuggingFace vision processors (CLIP, SigLIP, etc.)
    Provides a unified interface for vision model preprocessing.
    """
    
    def __init__(self, processor):
        """
        Args:
            processor: HuggingFace vision processor (e.g., CLIPImageProcessor)
        """
        self.processor = processor
    
    def __call__(self, images: Union[Image.Image, torch.Tensor], return_tensors: str = "pt"):
        """
        Process images for vision model.
        
        Args:
            images: PIL Image or torch.Tensor
            return_tensors: Return type ("pt" for PyTorch)
            
        Returns:
            Processed image tensor or BatchEncoding
        """
        if isinstance(images, torch.Tensor):
            # Already a tensor, assume it's preprocessed
            return images
        
        return self.processor(images=images, return_tensors=return_tensors)


__all__ = ["VisionProcessorWrapper"]
