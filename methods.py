import torch
import torch.nn.functional as F
import torch.nn as nn

def rotate_half(x):
    return torch.cat([-x[..., ::2], x[..., 1::2]], dim=-1)

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = sinu_pos[:q.size(1)]
    # print(q.shape, k.shape, sinu_pos.shape, rotate_half(q).shape)
    return (q * sinu_pos.cos()) + (rotate_half(q) * sinu_pos.sin()[None]), (k * sinu_pos.cos()) - (rotate_half(k) * sinu_pos.sin())

def get_sinu_pos_emb(dim, length):
    inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(0, length).float()
    sinu_pos = torch.einsum('i,j->ij', pos, inv_freq)
    emb = torch.cat((sinu_pos.sin(), sinu_pos.cos()), dim=-1)
    return emb

class RotaryEmbeddingTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(RotaryEmbeddingTransformerLayer, self).__init__()

        self.WQKV = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_p = dropout
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Norm first
        src = self.norm1(src)

        QKV = self.WQKV(src)
        p = QKV.shape[-1] // 3
        Q, K, V = QKV[..., :p], QKV[..., p:2*p], QKV[..., 2*p:]

        # Apply rotary embeddings to Q and K
        sinu_pos = get_sinu_pos_emb(src.size(-1), src.size(1)).to(src.device)
        Q, K = apply_rotary_pos_emb(Q, K, sinu_pos)

        # Self attention with causal mask
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True)
        #src = src + self.dropout1(attn_output)
        src = src + self.out_proj(attn_output)

        # Norm first for feed-forward network
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout1(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class RotaryEmbeddingTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(RotaryEmbeddingTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([RotaryEmbeddingTransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src, mask, is_causal):
        # Create a causal mask for src
        output = src
        for layer in self.layers:
            output = layer(output)
        return output

