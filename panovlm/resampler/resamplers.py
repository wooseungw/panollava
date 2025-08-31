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

class Conv1DResampler(BaseResampler):
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int):
        self.num_tokens = num_tokens
        super().__init__(in_dim, out_dim)
    
    def _build_layers(self):
        self.weight = nn.Parameter(torch.randn(self.num_tokens, self.in_dim))
        self.proj = nn.Linear(self.in_dim, self.out_dim)
    
    def forward(self, x):
        w = torch.softmax(self.weight, dim=0)
        mixed = (w.unsqueeze(0) * x).sum(dim=1, keepdim=True)
        return self.proj(mixed)

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
