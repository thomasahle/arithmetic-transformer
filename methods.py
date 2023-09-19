import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os


class ChannelDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ChannelDropout, self).__init__()
        self.p = p

    def forward(self, x):
        x_reshaped = x.reshape((-1,) + x.shape[-2:])
        x_dropped = F.dropout1d(x_reshaped, self.p, self.training, inplace=True)
        return x_dropped.reshape(*x.shape)

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

        if os.environ.get('DROP_MODE') in ('channel', 'head'):
            self.dropout_p = 0
            self.Q_dropout = ChannelDropout(dropout)
            self.K_dropout = ChannelDropout(dropout)
        if os.environ.get('DROP_MODE') in ('head', 'head-only'):
            self.head_dropout = ChannelDropout(dropout)

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

        # Try channel wise dropout instead of normal transformer dropout
        if os.environ.get('DROP_MODE') in ('channel', 'head'):
            Q = self.Q_dropout(Q)
            K = self.K_dropout(K)

        # Self attention with causal mask
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True)
        # Combine heads by swapping head and sequence channels
        attn_output = attn_output.permute(0, 2, 1, 3)
        # Head dropout!
        if os.environ.get('DROP_MODE') in ('head', 'head-only'):
            attn_output = self.head_dropout(attn_output)
        # Comibne head and head dimension
        attn_output = attn_output.flatten(2, 3)
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
            # Initialize to be roughly what ALiBi sets them as
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
        #mask = (torch.arange(seq)[None] + torch.arange(seq)[:, None])
        mask = torch.arange(seq)[None] + torch.arange(seq-1, -1, -1)[:, None] - (seq-1)
        mask = mask.to(x.device)

        match os.environ.get('ALIBI_METHOD'):
            case 'exp':
                mask = mask[None] * torch.exp(self.ms)[:, None, None]
                if random.random() < 1e-3:
                    print(self.level, torch.exp(self.ms).sort().values.round(decimals=3).detach())
            case 'sigmoid':
                mask = mask[None] * torch.sigmoid(self.ms)[:, None, None]
                if random.random() < 1e-3:
                    print(self.level, torch.sigmoid(self.ms).sort().values.round(decimals=3).detach())
            case 'softmax':
                mask = mask[None] * torch.softmax(self.ms, 0)[:, None, None]
                if random.random() < 1e-3:
                    print(self.level, torch.softmax(self.ms, 0).sort().values.round(decimals=3).detach())
            case 'single':
                mask = mask[None] * (2 ** -(torch.arange(self.num_heads, device=x.device) * F.softplus(self.m1)))[:, None, None]
                if random.random() < 1e-3:
                    print(self.level, 2**F.softplus(self.m1).item())
            case _:
                # Normal alibi exponential approach
                mask = mask[None] * (2 ** -torch.arange(self.num_heads))[:, None, None].to(x.device)

        # This is the normal alibi method, of allowing 0 on the diagonal
        triu = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(diagonal=1)

        self.mask = mask.float().masked_fill(triu, float('-inf'))
        if random.random() < 1e-3:
            print(self.mask)

    def forward(self, src):
        bs, seq, dim = src.shape

        # Norm first
        QKV = self.WQKV(self.norm(src))
        # Each of Q, K, V are now shaped (bs, head, seq, dim)
        Q, K, V = QKV.reshape(bs, seq, 3, self.num_heads, dim//self.num_heads).permute(2, 0, 3, 1, 4)

        self.mask = None
        self.ensure_mask_like(src)
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, attn_mask=self.mask)

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

