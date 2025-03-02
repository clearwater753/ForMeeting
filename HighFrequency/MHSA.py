import torch
import torch.nn as nn
import math

class MHSA(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.q_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask):
        # q: (N, q_len, embed_size), k: (N, k_len, embed_size), v: (N, v_len, embed_size)
        N = q.shape[0]
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        # q: (N, q_len, embed_size) -> (N, heads, q_len, head_dim)
        q = q.reshape(N, q_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, v_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3)/ math.sqrt(self.head_dim)
        # mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        # score: (N, heads, q_len, k_len)
        score = torch.softmax(score, dim=3)
        out = (score @ v).permute(0, 2, 1, 3).reshape(N, q_len, self.embed_size)
        out = self.fc_out(out) 
        return out

# Test
embed_size = 256
heads = 8
Attention = MHSA(embed_size, heads)
x = torch.rand((2, 100, embed_size))
mask = None
out = Attention(x, x, x, mask)
print(out.shape)
