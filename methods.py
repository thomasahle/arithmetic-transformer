import torch
import torch.nn.functional as F
import torch.nn as nn

def apply_rope(src, cos, sin):
    # Apply simply rotation matrix to every pair of values.
    x = src[..., 0::2]
    y = src[..., 1::2]
    out = torch.empty_like(src)
    out[..., 0::2] = x * cos - y * sin
    out[..., 1::2] = x * sin + y * cos
    return out

@torch.no_grad()
def make_cos_sin_like(x):
    length, dim = x.shape
    inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(0, length).float()
    outer = torch.outer(pos, inv_freq).to(x.device)
    return outer.cos(), outer.sin()

class RotaryEmbeddingTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(RotaryEmbeddingTransformerLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.WQKV = nn.Linear(d_model, 3 * d_model)
        self.dropout_p = dropout
        self.out_proj = nn.Linear(d_model, d_model)

        self.cos_sin = None

        # Two-layer MLP
        self.ffw = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src):
        bs, seq, dim = src.shape

        # Norm first
        QKV = self.WQKV(self.norm(src))
        p = QKV.shape[-1] // 3
        Q, K, V = QKV[..., :p], QKV[..., p:2*p], QKV[..., 2*p:]

        # Apply rotary embeddings to Q and K
        if self.cos_sin is None or self.cos_sin[0].shape[0] != seq:
            self.cos_sin = make_cos_sin_like(src[0])
        #sinu_pos = get_sinu_pos_emb(src.size(-1), src.size(1)).to(src.device)
        #Q, K = apply_rotary_pos_emb(Q, K, sinu_pos)
        Q = apply_rope(Q, *self.cos_sin)
        K = apply_rope(K, *self.cos_sin)

        # Self attention with causal mask
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True)
        src = src + self.out_proj(attn_output)

        # Norm first for feed-forward network
        src = src + self.ffw(src)

        return src


class RotaryEmbeddingTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(RotaryEmbeddingTransformerEncoder, self).__init__()
        self.layers = nn.Sequential(*[
            RotaryEmbeddingTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])

    def forward(self, src, mask, is_causal):
        # No need to be residual here, since the layers themselves are residual.
        return self.layers(src)

