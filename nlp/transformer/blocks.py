import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_size, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, embed_size, device=device, requires_grad=False)
        pos = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        _2i = torch.arange(0, embed_size, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))
    
    def forward(self, x):
        return self.encoding[:x.shape[1], :]
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, drop_prob, device):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size)
        self.positional_embedding = PositionalEmbedding(max_len, embed_size, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.device = device
        self.embed_size = embed_size
    
    def forward(self, x):
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(x)
        return self.drop_out(tok_emb + pos_emb)

class LayerNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-10):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))
        self.eps = eps
    
    def forward(self, x):
        # x: (N, seq_len, embed_size)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
                                    
class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_size, hidden, drop_prob):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden)
        self.fc2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        # x: (N, seq_len, embed_size)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

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

    def forward(self, q, k, v, mask=None):
        # q: (N, q_len, embed_size), k: (N, k_len, embed_size), v: (N, v_len, embed_size)
        N = q.shape[0]
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        # q: (N, q_len, embed_size) -> (N, heads, q_len, head_dim)
        q = q.reshape(N, q_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, v_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3)/ math.sqrt(self.head_dim)
        if mask is not None:
            # mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool))
            score = score.masked_fill(mask == 0, float("-inf"))
        # score: (N, heads, q_len, k_len)
        score = torch.softmax(score, dim=3)
        out = (score @ v).permute(0, 2, 1, 3).reshape(N, q_len, self.embed_size)
        out = self.fc_out(out) 
        return out

class Encoder_layer(nn.Module):
    def __init__(self, embed_size, heads, ffn_hidden, drop_prob):
        super().__init__()
        self.mhsa = MHSA(embed_size, heads)
        self.drop1 = nn.Dropout(drop_prob)
        self.norm1 = LayerNorm(embed_size)
        self.ffn = PositionwiseFeedforward(embed_size, ffn_hidden, drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.norm2 = LayerNorm(embed_size)
    
    def forward(self, x, mask):
        # x: (N, seq_len, embed_size)
        x = self.norm1(x + self.drop1(self.mhsa(x, x, x, mask)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x

class Decoder_layer(nn.Module):
    def __init__(self, embed_size, heads, ffn_hidden, drop_prob):
        super().__init__()
        self.mhsa = MHSA(embed_size, heads)
        self.drop1 = nn.Dropout(drop_prob)
        self.norm1 = LayerNorm(embed_size)

        self.cross_mhsa = MHSA(embed_size, heads)
        self.drop2 = nn.Dropout(drop_prob)
        self.norm2 = LayerNorm(embed_size)

        self.ffn = PositionwiseFeedforward(embed_size, ffn_hidden, drop_prob)
        self.drop3 = nn.Dropout(drop_prob)
        self.norm3 = LayerNorm(embed_size)

    def forward(self, enc, dec, p_mask, mask):
        # p_mask: padding mask, mask: look ahead mask
        # x: (N, seq_len, embed_size)
        x = self.norm1(dec + self.drop1(self.mhsa(dec, dec, dec, mask)))
        x = self.norm2(x + self.drop2(self.cross_mhsa(x, enc, enc, p_mask)))
        x = self.norm3(x + self.drop3(self.ffn(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, embed_size, num_layers, heads, ffn_hidden, drop_prob, device, max_len):
        super().__init__()
        self.embedding = TransformerEmbedding(enc_vocab_size, embed_size, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                Encoder_layer(embed_size, heads, ffn_hidden, drop_prob) 
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, p_mask):
        # x: (N, seq_len, embed_size)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, p_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, embed_size, num_layers, heads, ffn_hidden, drop_prob, device, max_len):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_vocab_size, embed_size, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                Decoder_layer(embed_size, heads, ffn_hidden, drop_prob) 
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, dec_vocab_size)

    def forward(self, enc, dec, p_mask, mask):
        # x: (N, seq_len, embed_size)
        dec = self.embedding(dec)
        for layer in self.layers:
            x = layer(enc, dec, p_mask, mask)
        return self.fc(x)