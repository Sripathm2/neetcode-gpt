import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.model_dim = model_dim

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        q = self.q_proj(x)
        new_k = self.k_proj(x)
        new_v = self.v_proj(x)

        if kv_cache is None:
            kv_cache = KVCache()

        prev_len = 0 if kv_cache.cache_k is None else kv_cache.cache_k.shape[1]

        K, V = kv_cache.update(new_k, new_v)

        model_dim = q.shape[-1]
        scores = (q @ K.transpose(-2, -1)) / math.sqrt(model_dim)

        new_len = q.shape[1]
        total_len = K.shape[1]
        row_idx = torch.arange(new_len).unsqueeze(1) + prev_len   # (new_len, 1)
        col_idx = torch.arange(total_len).unsqueeze(0)            # (1, total_len)
        mask = col_idx <= row_idx    # True where key position is allowed (causal)

        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)

        output = attn_weights @ V

        return torch.round(output, decimals=4), kv_cache
