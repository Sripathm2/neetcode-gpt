import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        attn_score = Q @ torch.transpose(K,1,2)
        context_length, attention_dim = K.shape[1], K.shape[2]
        attn_score = attn_score / (attention_dim ** 0.5)

        lower_triangle = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangle == 0
        attn_score = attn_score.masked_fill(mask, float('-inf'))

        attn_score = torch.softmax(attn_score, dim=2)

        return torch.round(attn_score @ V, decimals=4)

