from typing import Optional, Union
import torch 
from torch import nn
from transformers import (AutoModel, AutoConfig, AutoModelForCausalLM,
                          BertConfig, BertModel, BatchEncoding)
from transformers.modeling_outputs import BaseModelOutput

# ===============================================================
# 0.  Base Resampler Class
# ===============================================================
class BaseResampler(nn.Module):
    """
    기본 리샘플러 클래스 - 입출력 차원 변경의 기본 구조 제공
    
    모든 리샘플러는 입력 차원(in_dim)을 출력 차원(out_dim)으로 변환하는 역할을 수행합니다.
    
    Args:
        in_dim: 입력 특성 차원
        out_dim: 출력 특성 차원
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._build_layers()
    
    def _build_layers(self):
        """서브클래스에서 구현할 레이어 구성 메서드"""
        raise NotImplementedError("Subclasses must implement _build_layers()")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 텐서의 마지막 차원을 in_dim에서 out_dim으로 변환
        
        Args:
            x: 입력 텐서 [..., in_dim]
            
        Returns:
            출력 텐서 [..., out_dim]
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    @property
    def config(self):
        """리샘플러 설정 정보 반환"""
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
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
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
        # Input projection to match dimensions
        self.input_proj = nn.Linear(self.in_dim, self.out_dim) if self.in_dim != self.out_dim else nn.Identity()
        
        # ConvNeXt stages - repeated num_repeats times
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
        
        # Global average pooling and final projection
        self.norm = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.head = nn.Linear(self.out_dim, self.out_dim)
    
    def forward(self, x):
        # x: (B, seq_len, in_dim)
        B, seq_len, _ = x.shape
        
        # Project input dimensions
        x = self.input_proj(x)  # (B, seq_len, out_dim)
        
        # Reshape to 2D format for ConvNeXt: create spatial dimensions
        H = W = int(seq_len ** 0.5) if seq_len == int(seq_len ** 0.5) ** 2 else int(seq_len ** 0.5) + 1
        pad_len = H * W - seq_len
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, self.out_dim, device=x.device)], dim=1)
        
        x = x.view(B, H, W, self.out_dim).permute(0, 3, 1, 2)  # (B, out_dim, H, W)
        
        # Apply ConvNeXt blocks repeatedly
        for repeat_stages in self.repeated_stages:
            for stage in repeat_stages:
                for block in stage:
                    x = block(x)
        
        # Global average pooling and reshape back
        x = x.permute(0, 2, 3, 1)  # (B, H, W, out_dim)
        x = x.view(B, H * W, self.out_dim)  # (B, H*W, out_dim)
        
        # Remove padding if added
        if pad_len > 0:
            x = x[:, :seq_len, :]
        
        # Global average pooling to get fixed number of tokens
        x = self.norm(x)
        x = x.mean(dim=1, keepdim=True)  # (B, 1, out_dim)
        x = self.head(x)
        
        return x

class QFormerResampler(BaseResampler):
    """Mini Q-Former: learnable query tokens + BERT encoder layer stack."""
    def __init__(self, in_dim: int, out_dim: int, num_query: int = 32, num_hidden_layers: int = 6):
        self.num_query = num_query
        self.num_hidden_layers = num_hidden_layers
        super().__init__(in_dim, out_dim)
    
    def _build_layers(self):
        cfg = BertConfig(hidden_size=self.in_dim, num_hidden_layers=self.num_hidden_layers,
                         num_attention_heads=8, intermediate_size=self.in_dim*4,
                         is_decoder=False, add_cross_attention=True)
        self.bert = BertModel(cfg)
        self.query = nn.Parameter(torch.randn(1, self.num_query, self.in_dim))
        self.proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):                        # x: (B, N, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)         # (B,Q,D)
        out = self.bert(inputs_embeds=q, encoder_hidden_states=x,
                         encoder_attention_mask=torch.ones(x.size()[:-1], device=x.device))
        return self.proj(out.last_hidden_state)  # (B, Q, out_dim)


# ===============================================================
# 2.  MLP Resampler (moved from panovlm/model.py)
# ===============================================================
class MLPResampler(BaseResampler):
    """
    Simple MLP Resampler: vision_dim → latent_dim

    Args:
        vision_dim: 입력 차원 (vision encoder의 hidden_size)
        latent_dim: 출력 차원 (language model로 전달될 차원)
        hidden_dim: 중간 레이어 차원 (기본값: max(vision_dim, latent_dim))
        depth: MLP 깊이
        use_ln: LayerNorm 사용 여부
    """
    def __init__(self, vision_dim: int, latent_dim: int, hidden_dim: Optional[int] = None,
                 depth: int = 3, use_ln: bool = True):
        self.vision_dim = vision_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_ln = use_ln
        super().__init__(vision_dim, latent_dim)
    
    def _build_layers(self):
        # 중간 레이어 차원: 기본적으로 입력/출력 차원 중 큰 값 사용
        hidden_dim = self.hidden_dim or max(self.in_dim, self.out_dim)

        # MLP 구성
        layers = []
        current_dim = self.in_dim

        # 중간 레이어들
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_ln:
                layers.append(nn.LayerNorm(hidden_dim, eps=1e-5))
            layers.append(nn.GELU())
            current_dim = hidden_dim

        # 최종 출력 레이어
        layers.append(nn.Linear(current_dim, self.out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B*V, S, vision_dim]
        Returns:
            resampled: [B*V, S, latent_dim]
        """
        BV, S, _ = vision_features.shape

        # MLP 적용: token-wise transformation
        x = vision_features.reshape(-1, self.in_dim)  # [B*V*S, in_dim]
        x = self.mlp(x)  # [B*V*S, out_dim]
        x = x.view(BV, S, self.out_dim)  # [B*V, S, out_dim]

        return x
