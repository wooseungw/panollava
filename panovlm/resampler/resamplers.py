from typing import Optional, Union
import torch 
from torch import nn
from transformers import (AutoModel, AutoConfig, AutoModelForCausalLM,
                          BertConfig, BertModel, BatchEncoding)
from transformers.modeling_outputs import BaseModelOutput

# ===============================================================
# 1.  Generic Resampler Blocks + QFormerResampler
# ===============================================================
class IdentityResampler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(); self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)

class AvgPoolResampler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(); self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x.mean(dim=1, keepdim=True))

class Conv1DResampler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_tokens, in_dim))
        self.proj   = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        w = torch.softmax(self.weight, dim=0)
        mixed = (w.unsqueeze(0) * x).sum(dim=1, keepdim=True)
        return self.proj(mixed)

class QFormerResampler(nn.Module):
    """Mini Q-Former: learnable query tokens + BERT encoder layer stack."""
    def __init__(self, in_dim: int, out_dim: int, num_query: int = 32, num_hidden_layers: int = 6):
        super().__init__()
        cfg = BertConfig(hidden_size=in_dim, num_hidden_layers=num_hidden_layers,
                         num_attention_heads=8, intermediate_size=in_dim*4,
                         is_decoder=False, add_cross_attention=True)
        self.bert = BertModel(cfg)
        self.query = nn.Parameter(torch.randn(1, num_query, in_dim))
        self.proj  = nn.Linear(in_dim, out_dim)

    def forward(self, x):                        # x: (B, N, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)         # (B,Q,D)
        out = self.bert(inputs_embeds=q, encoder_hidden_states=x,
                         encoder_attention_mask=torch.ones(x.size()[:-1], device=x.device))
        return self.proj(out.last_hidden_state)  # (B, Q, out_dim)