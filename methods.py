import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os

def make_ffw(d_model, dim_feedforward, dropout):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
        nn.Dropout(dropout),
    )

class RotaryEmbeddingTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        assert d_model % num_heads == 0

        self.norm = nn.LayerNorm(d_model)
        self.WQKV = nn.Linear(d_model, 3 * d_model)
        self.dropout_p = dropout
        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_model, d_model)

        # Cached rope mask
        self.cos_sin = None

        # Two-layer MLP unless dim_feedforward is 0, in which case we make an "attention only"
        # transformer.
        if dim_feedforward != 0:
            self.ffw = make_ffw(d_model, dim_feedforward, dropout)
        else:
            self.ffw = None


    def apply_rope(self, src):
        self.ensure_cos_sin_like(src)
        cos, sin = self.cos_sin
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
    def ensure_cos_sin_like(self, x):
        seq, dim = x.shape[-2: ]
        # Test if we already have a cos_sin of the right size
        if self.cos_sin is not None and self.cos_sin[0].shape == (seq, dim):
            return
        # Outer product of m * theta^(1000/k)
        outer = torch.outer(
                torch.arange(0, seq),
                1. / (10000 ** (torch.arange(0, dim, 2) / dim))
            ).to(x.device)
        self.cos_sin = (outer.cos(), outer.sin())

    def forward(self, src):
        bs, seq, dim = src.shape

        # Norm first
        QKV = self.WQKV(self.norm(src))
        # Each of Q, K, V are now shaped (bs, head, seq, dim)
        Q, K, V = QKV.reshape(bs, seq, 3, self.num_heads, dim//self.num_heads).permute(2, 0, 3, 1, 4)

        # Apply rotary embeddings to Q and K
        Q = self.apply_rope(Q)
        K = self.apply_rope(K)

        # Self attention with causal mask
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True)
        # Combine heads
        attn_output = attn_output.permute(0, 2, 1, 3).flatten(2, 3)
        src = src + self.out_proj(attn_output)

        # Norm first for feed-forward network
        if self.ffw is not None:
            src = src + self.ffw(src)

        return src

class AlibiTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, level):
        super().__init__()
        assert d_model % num_heads == 0

        self.norm = nn.LayerNorm(d_model)
        self.WQKV = nn.Linear(d_model, 3 * d_model)
        self.dropout_p = dropout
        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_model, d_model)

        self.ms = nn.Parameter(torch.empty(num_heads))
        self.m1 = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.ms[:] = - torch.arange(0, num_heads)

        self.level = level

        # Cached rope mask
        self.mask = None

        # Two-layer MLP unless dim_feedforward is 0, in which case we make an "attention only"
        # transformer.
        if dim_feedforward != 0:
            self.ffw = make_ffw(d_model, dim_feedforward, dropout)
        else:
            self.ffw = None

    def ensure_mask_like(self, x):
        seq = x.shape[-2]
        if self.mask is not None and self.mask.shape[-1] == seq:
            return
        mask = (torch.arange(seq)[None] + torch.arange(seq)[:, None]).to(x.device)

        if os.environ['ALIBI_METHOD'] == 'normal':
            mask = - mask[None] * (2 ** -torch.arange(self.num_heads))[:, None, None].to(x.device)
        elif os.environ['ALIBI_METHOD'] == 'exp':
            mask = - mask[None] * torch.exp(self.ms)[:, None, None]
            if random.random() < 1e-3:
                print(self.level, torch.exp(self.ms).sort().values.round(decimals=3).detach())
        elif os.environ['ALIBI_METHOD'] == 'sigmoid':
            mask = - mask[None] * torch.sigmoid(self.ms)[:, None, None]
            if random.random() < 1e-3:
                print(self.level, torch.sigmoid(self.ms).sort().values.round(decimals=3).detach())
        elif os.environ['ALIBI_METHOD'] == 'softmax':
            mask = - mask[None] * torch.softmax(self.ms, 0)[:, None, None]
            if random.random() < 1e-3:
                print(self.level, torch.softmax(self.ms, 0).sort().values.round(decimals=3).detach())
        elif os.environ['ALIBI_METHOD'] == 'single':
            mask = - mask[None] * (2 ** -(torch.arange(self.num_heads, device=x.device) * F.softplus(self.m1)))[:, None, None]
            if random.random() < 1e-3:
                print(self.level, 2**F.softplus(self.m1).item())

        # Seems higher levels should be less local, but it doesn't seem to work for me.
        #mask = mask[None] * (- 2. ** -torch.arange(self.num_heads))[:, None, None].to(x.device)
        # mask /= float(self.level + 1)

        triu = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(diagonal=1)
        mask = mask.float().to(x.device)
        self.mask = mask.masked_fill(triu, float('-inf'))

    # normal
    # softmax
    # sigmoid
    # raw weighted

    def forward(self, src):
        bs, seq, dim = src.shape

        # Norm first
        QKV = self.WQKV(self.norm(src))
        # Each of Q, K, V are now shaped (bs, head, seq, dim)
        Q, K, V = QKV.reshape(bs, seq, 3, self.num_heads, dim//self.num_heads).permute(2, 0, 3, 1, 4)

        self.mask = None
        self.ensure_mask_like(src)
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, attn_mask=self.mask)
        # attn_output = my_scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, attn_mask=self.mask)

        # Recombine heads
        src = src + self.out_proj(attn_output.permute(0, 2, 1, 3).flatten(2, 3))
        # Norm first for feed-forward network
        if self.ffw is not None:
            src = src + self.ffw(src)
        return src

class RNNTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        assert d_model % num_heads == 0

        self.norm = nn.LayerNorm(d_model)
        dh = d_model // num_heads
        self.Qrnn = nn.RNN(dh, dh, batch_first=True)
        self.Krnn = nn.RNN(dh, dh, batch_first=True)
        self.WQKV = nn.Linear(d_model, 3 * d_model)
        self.dropout_p = dropout
        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_model, d_model)

        # Cached rope mask
        self.mask = None

        self.ffw = make_ffw(d_model, dim_feedforward, dropout)

    def forward(self, src):
        bs, seq, dim = src.shape

        # Norm first
        QKV = self.WQKV(self.norm(src))
        # Each of Q, K, V are now shaped (bs, head, seq, dim)
        Q, K, V = QKV.reshape(bs, seq, 3, self.num_heads, dim//self.num_heads).permute(2, 0, 3, 1, 4)

        #self.ensure_mask_like(src)
        Q = self.Qrnn(Q)[0]
        K = self.Krnn(K)[0]
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True)

        # Recombine heads
        src = src + self.out_proj(attn_output.permute(0, 2, 1, 3).flatten(2, 3))
        # Norm first for feed-forward network
        src = src + self.ffw(src)
        return src

