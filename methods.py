import torch
import torch.nn.functional as F
import torch.nn as nn

def apply_rope(src, cos, sin):
    # Apply simply rotation matrix to every pair of values.
    # Even/odd is better than splitting "first/last" half, as neighbouring
    # values will go into the same attention head.
    x = src[..., 0::2]
    y = src[..., 1::2]
    out = torch.empty_like(src)
    out[..., 0::2] = x * cos - y * sin
    out[..., 1::2] = x * sin + y * cos
    return out

@torch.no_grad()
def make_cos_sin_like(x):
    length, dim = x.shape[-2:]
    inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2) / dim))
    pos = torch.arange(0, length)
    # Outer product of m * theta^(1000/k)
    outer = torch.outer(pos, inv_freq).to(x.device)
    return outer.cos(), outer.sin()

class RotaryEmbeddingTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super(RotaryEmbeddingTransformerLayer, self).__init__()
        assert d_model % num_heads == 0

        self.norm = nn.LayerNorm(d_model)
        self.WQKV = nn.Linear(d_model, 3 * d_model)
        self.dropout_p = dropout
        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_model, d_model)

        self.cos_sin = None

        # Two-layer MLP unless dim_feedforward is 0, in which case we make an "attention only"
        # transformer.
        self.ffw = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        ) if dim_feedforward != 0 else None

    def forward(self, src):
        bs, seq, dim = src.shape

        # Norm first
        QKV = self.WQKV(self.norm(src))
        # Each of Q, K, V are now shaped (bs, head, seq, dim)
        Q, K, V = QKV.reshape(bs, seq, 3, self.num_heads, dim//self.num_heads).permute(2, 0, 3, 1, 4)

        # Apply rotary embeddings to Q and K
        if self.cos_sin is None or self.cos_sin[0].shape[0] != seq:
            self.cos_sin = make_cos_sin_like(Q)
        Q = apply_rope(Q, *self.cos_sin)
        K = apply_rope(K, *self.cos_sin)

        # Self attention with causal mask
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True)
        # Combine heads
        attn_output = attn_output.permute(0, 2, 1, 3).flatten(2, 3)
        src = src + self.out_proj(attn_output)

        # Norm first for feed-forward network
        if self.ffw is not None:
            src = src + self.ffw(src)

        return src


class RotaryEmbeddingTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(RotaryEmbeddingTransformerEncoder, self).__init__()
        self.layers = nn.Sequential(*[
            RotaryEmbeddingTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])

    def forward(self, src, mask=None, is_causal=True):
        assert is_causal
        # No need to be residual here, since the layers themselves are residual.
        return self.layers(src)

