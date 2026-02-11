import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SphereUformerConfig:
    hidden_size: int = 768
    model_type: str = "sphere_uformer"

class SphereUformer(nn.Module):
    """
    Placeholder implementation of SphereUformer for PanoramaVLM integration.
    
    This class is intended to be loaded via VisionBackbone with backbone_type="module".
    It simulates a vision encoder that processes spherical inputs.
    """
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__()
        self.config = SphereUformerConfig(hidden_size=hidden_size)
        self.hidden_size = hidden_size
        
        # Dummy layers to simulate processing and ensure gradients can flow
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
    def forward(self, pixel_values, **kwargs):
        """
        Args:
            pixel_values: Tensor of shape (B, C, H, W) or (B, V, C, H, W) flattened
            
        Returns:
            Tensor of shape (B, SeqLen, HiddenSize)
        """
        # VisionBackbone flattens (B, V) into B', so we expect (B', C, H, W)
        # or it might pass (B', H, W, C) depending on dataset. 
        # But standard PyTorch convention is (B, C, H, W).
        
        # Handle potential input shapes
        if pixel_values.ndim == 4:
            # (B, C, H, W) - Standard
            pass
        elif pixel_values.ndim == 3:
             # (B, Seq, Dim) - Already features? Unlikely for "pixel_values"
             pass
             
        # Simple patch embedding
        # x: (B, C, H, W) -> (B, D, H', W')
        x = self.patch_embed(pixel_values)
        
        # Flatten spatial dimensions
        # (B, D, H', W') -> (B, D, N) -> (B, N, D)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Pass through transformer
        x = self.encoder(x)
        
        return x

    def forward_features(self, pixel_values):
        return self.forward(pixel_values)
