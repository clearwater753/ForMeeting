import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # query shape: (N, query_len, embed_size)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        queries = query.reshape(N, query_len, self.heads, self.head_dim).transpose(1, 2)
        # (N, heads, query_len, head_dim) @ (N, heads, head_dim, key_len) -> (N, heads, query_len, key_len)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        energy = queries@keys.transpose(-1, -2)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        # attention shape: (N, heads, query_len, key_len)
        attention = torch.softmax( energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention@values shape: (N, heads, query_len, head_dim)
        
        out = (attention@values).transpose(1, 2).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out) 
        return out

# Example usage:
embed_size = 256
heads = 8
attention_layer = SelfAttention(embed_size, heads)

# Dummy data
N, value_len, key_len, query_len = 3, 40, 40, 30
value = torch.rand((N, value_len, embed_size))
key = torch.rand((N, key_len, embed_size))
query = torch.rand((N, query_len, embed_size))
mask = None  # Optional mask for padded tokens

# Forward pass
out = attention_layer(value, key, query, mask)
print(out.shape)  # Should be (N, query_len, embed_size)