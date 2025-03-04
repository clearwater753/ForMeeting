import torch 
import torch.nn as nn
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_size, heads, groups):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.groups = groups

        assert embed_size % heads == 0, "Heads must be divisible by groups"
        self.group_heads = heads // groups
        self.head_dim = embed_size // heads

        self.q_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.k_proj = nn.Linear(self.embed_size, groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_size, groups * self.head_dim, bias=False)

        self.fc = nn.Linear(self.embed_size, self.embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def expand(self, x):
        N, seq_len = x.shape[0], x.shape[2]
        x = x[:, :, None, :, :].expand(N, self.groups, self.group_heads, seq_len, self.head_dim).contiguous()
        return x.view(N, self.groups*self.group_heads, seq_len, self.head_dim)


    def forward(self, q, k, v, mask=None):
        # k: [N, k_len, embed_size] -> [N, k_len, groups * head_dim]
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        N = q.shape[0]
        q = q.view(N, -1, self.groups * self.group_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(N, -1, self.groups, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(N, -1, self.groups, self.head_dim).permute(0, 2, 1, 3)

        k, v = self.expand(k), self.expand(v)

        score = q @ k.tarnspose(2, 3) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        score = self.softmax(score)
        out = score @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(N, -1, self.embed_size)
        out = self.fc(out)
        return out