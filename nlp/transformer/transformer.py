from blocks import *
import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, src_pad_idx,
                trg_pad_idx,
                enc_vocab_size,
                dec_vocab_size, 
                embed_size, 
                num_layers, 
                heads, 
                ffn_hidden, 
                drop_prob, 
                device, 
                max_len):
        super().__init__()
        self.encoder = Encoder(enc_vocab_size, embed_size, num_layers, heads, ffn_hidden, drop_prob, device, max_len)
        self.decoder = Decoder(dec_vocab_size, embed_size, num_layers, heads, ffn_hidden, drop_prob, device, max_len)
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
    
    def make_casual_mask(self, q, k):
        q_len, k_len = q.shape[1], k.shape[1]
        mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool)).to(self.device)
        return mask

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        # q: (N, q_len), k: (N, k_len)
        len_q, len_k = q.size(1), k.size(1)

        # (Batch, 1, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(
            trg, trg, self.trg_pad_idx, self.trg_pad_idx
        ) * self.make_casual_mask(trg, trg)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        ouput = self.decoder(enc, trg, trg_mask, src_trg_mask)
        return ouput

enc_voc_size = 6000
dec_voc_size = 8000
src_pad_idx = 1
trg_pad_idx = 1
trg_sos_idx = 2
batch_size = 128
max_len = 1024
embed_size = 512
n_layers = 3
n_heads = 2
ffn_hidden = 1024
drop_prob = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    embed_size=embed_size,
                    enc_vocab_size=enc_voc_size,
                    dec_vocab_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    heads=n_heads,
                    num_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
        
model.apply(initialize_weights)
src = torch.load('./nlp/transformer/tensor_src.pt')
src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int)), dim=-1)
trg = torch.load('./nlp/transformer/tensor_trg.pt')
result = model(src, trg)
print(src.shape, trg.shape)
print(result.shape)