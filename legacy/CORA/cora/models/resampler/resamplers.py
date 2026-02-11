from typing import Optional, Union, List
import torch
from torch import nn
from transformers import BertConfig, BertModel

import logging

logger = logging.getLogger(__name__)

# ===============================================================
# 0.  Base Resampler Class
# ===============================================================
class BaseResampler(nn.Module):
    """
    Base Resampler Class - Provides structure for dimension transformation.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._build_layers()
    
    def _build_layers(self):
        """Subclasses must implement this to build layers."""
        raise NotImplementedError("Subclasses must implement _build_layers()")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., in_dim]
        Returns:
            Output tensor [..., out_dim]
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    @property
    def config(self):
        return {
            'type': self.__class__.__name__,
            'in_dim': self.in_dim,
            'out_dim': self.out_dim
        }

# ===============================================================
# 1.  Generic Resampler Blocks + QFormerResampler
# ===============================================================
class IdentityResampler(BaseResampler):
    def _build_layers(self):
        self.proj = nn.Linear(self.in_dim, self.out_dim)
    
    def forward(self, x):
        return self.proj(x)

class AvgPoolResampler(BaseResampler):
    def _build_layers(self):
        self.proj = nn.Linear(self.in_dim, self.out_dim)
    
    def forward(self, x):
        return self.proj(x.mean(dim=1, keepdim=True))

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted for 2D processing"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x

class ConvResampler(BaseResampler):
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int, depths=[2, 2], drop_path_rate=0., num_repeats=1):
        self.num_tokens = num_tokens
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.num_repeats = num_repeats
        super().__init__(in_dim, out_dim)
    
    def _build_layers(self):
        self.input_proj = nn.Linear(self.in_dim, self.out_dim) if self.in_dim != self.out_dim else nn.Identity()
        self.repeated_stages = nn.ModuleList()
        
        for repeat in range(self.num_repeats):
            dp_rates = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
            cur = 0
            stages = nn.ModuleList()
            for _, depth in enumerate(self.depths):
                stage = nn.ModuleList([
                    ConvNeXtBlock(dim=self.out_dim, drop_path=dp_rates[cur + j]) 
                    for j in range(depth)
                ])
                stages.append(stage)
                cur += depth
            self.repeated_stages.append(stages)
        
        self.norm = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.head = nn.Linear(self.out_dim, self.out_dim)
    
    def forward(self, x):
        B, seq_len, _ = x.shape
        x = self.input_proj(x)
        
        H = W = int(seq_len ** 0.5) if seq_len == int(seq_len ** 0.5) ** 2 else int(seq_len ** 0.5) + 1
        pad_len = H * W - seq_len
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, self.out_dim, device=x.device)], dim=1)
        
        x = x.view(B, H, W, self.out_dim).permute(0, 3, 1, 2)
        
        for repeat_stages in self.repeated_stages:
            for stage in repeat_stages:
                for block in stage:
                    x = block(x)
        
        x = x.permute(0, 2, 3, 1).view(B, H * W, self.out_dim)
        
        if pad_len > 0:
            x = x[:, :seq_len, :]
        
        x = self.norm(x)
        x = x.mean(dim=1, keepdim=True)
        return self.head(x)

class QFormerResampler(BaseResampler):
    """Mini Q-Former: learnable query tokens + BERT encoder layer stack."""
    def __init__(self, in_dim: int, out_dim: int, num_query: int = 32, num_hidden_layers: int = 6):
        self.num_query = num_query
        self.num_hidden_layers = num_hidden_layers
        super().__init__(in_dim, out_dim)
    
    def _build_layers(self):
        cfg = BertConfig(
            hidden_size=self.in_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=8,
            intermediate_size=self.in_dim * 4,
        )
        cfg.is_decoder = True
        cfg.add_cross_attention = True
        cfg.encoder_width = self.in_dim
        self.bert = BertModel(cfg)
        self.query = nn.Parameter(torch.randn(1, self.num_query, self.in_dim))
        self.proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out = self.bert(inputs_embeds=q, encoder_hidden_states=x,
                         encoder_attention_mask=torch.ones(x.size()[:-1], device=x.device))
        return self.proj(out.last_hidden_state)


# ===============================================================
# 2.  MLP Resampler
# ===============================================================
class MLPResampler(BaseResampler):
    def __init__(self, vision_dim: int, latent_dim: int, hidden_dim: Optional[int] = None,
                 depth: int = 3, use_ln: bool = True, pool_tokens: Optional[int] = None,
                 pool_type: str = "avg"):
        self.vision_dim = vision_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_ln = use_ln
        self.pool_tokens = pool_tokens
        self.pool_type = pool_type
        super().__init__(vision_dim, latent_dim)
    
    def _build_layers(self):
        hidden_dim = self.hidden_dim or max(self.in_dim, self.out_dim)
        layers = []
        current_dim = self.in_dim

        for _ in range(self.depth - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_ln:
                layers.append(nn.LayerNorm(hidden_dim, eps=1e-5))
            layers.append(nn.GELU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        BV, S, _ = vision_features.shape

        x = vision_features.reshape(-1, self.in_dim)
        x = self.mlp(x)
        x = x.view(BV, S, self.out_dim)

        if self.pool_tokens is not None and self.pool_tokens > 0 and self.pool_tokens < S:
            if self.pool_type == "max":
                x = torch.nn.functional.adaptive_max_pool1d(x.transpose(1, 2), self.pool_tokens).transpose(1, 2)
            else:
                x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.pool_tokens).transpose(1, 2)

        return x
