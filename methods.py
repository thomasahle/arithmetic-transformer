
import torch.nn as nn

class BayesLinear(nn.Module):
    def __init__(self, input_dim, output_dim, stdv=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_stdv = stdv / (input_dim + output_dim)**.5

        # Initialize weight, bias and stdv tensors
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.stdv = nn.Parameter(torch.Tensor(output_dim, input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for weights
        nn.init.kaiming_uniform_(self.weight)
        # Zero initialization for biases
        nn.init.zeros_(self.bias)
        # Fill stdv with a constant value of 0.1
        nn.init.constant_(self.stdv, self.initial_stdv)

    def forward(self, x):
        # Add noise to weights
        if self.training:
            #noise = torch.randn_like(self.weight)
            # Alternatively we could do +/-1 (faster):
            noise = torch.randint(2, self.weight.shape, device=self.weight.device) * 2 - 1
            weight_noised = self.weight + noise * self.stdv
        else:
            weight_noised = self.weight
        return F.linear(x, weight_noised, self.bias)

    # The real gaussian kl-penalty is actually
    # log s1/s2 + [s2^2 + (m1 - m2)^2]/s1^2
    # but we have s1=1 and m1 = 0, so this is just
    # log 1/s2 + s2^2 + (m1 - m2)^2,
    # up to a constant. And the square terms are covered by AdamW, so
    # we just have to handle the log1/stdv.
    def penalty(self):
        #return torch.mean(torch.log(1e-8 + torch.abs(1.0 / self.stdv)))
        return -torch.mean(torch.log(1e-8 + torch.abs(self.stdv / self.initial_stdv)))


class BayesLora(nn.Module):
    def __init__(self, input_dim, output_dim, rank, stdv=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_stdv = stdv / (input_dim + output_dim)**.5

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.std0 = nn.Parameter(torch.Tensor(rank, input_dim))
        self.std1 = nn.Parameter(torch.Tensor(output_dim, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        # Calculate the right initial value to make the product of
        # std0 and std1 match initial_stdv
        s = (self.initial_stdv / self.std0.shape[0])**.5
        nn.init.constant_(self.std0, s)
        nn.init.constant_(self.std1, s)

    def forward(self, x):
        if self.training:
            noise = torch.randint(2, self.weight.shape, device=self.weight.device) * 2 - 1
            weight_noised = self.weight + noise * (self.std1 @ self.std0)
        else:
            weight_noised = self.weight
        return F.linear(x, weight_noised, self.bias)

    def penalty(self):
        expand_noise = (self.std1 @ self.std0)
        return -torch.mean(torch.log(1e-8 + torch.abs(expand_noise / self.initial_stdv)))


class BayesLinearBias(nn.Module):
    def __init__(self, input_dim, output_dim, stdv=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_stdv = stdv / (input_dim + output_dim)**.5

        # Initialize weight, bias and stdv tensors
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.stdv = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.stdv, self.initial_stdv)

    def forward(self, x):
        if self.training:
            #noise = torch.randn_like(self.bias)
            # Alternatively we could do +/-1 (faster):
            noise = torch.randint(2, self.bias.shape, device=self.bias.device) * 2 - 1
            bias_noised = self.bias + noise * self.stdv
        else:
            bias_noised = self.bias
        return F.linear(x, self.weight, bias_noised)

    def penalty(self):
        # Becaue the bias is 1 dimensional, it doesn't automatically get weigth decay.
        # So we have to add it here ourselves.
        r = self.stdv / self.initial_stdv
        return -torch.sum(torch.log(1e-8 + torch.abs(r))) + torch.sum(r**2)


class GroupedAttention(nn.Module):
    def __init__(self, dim, n_queries, n_keys, is_cosine, norm_kvs, dropout=0):
        super().__init__()
        assert dropout == 0, "Dropout not supported yet"
        assert n_queries % n_keys == 0
        self.n_groups = n_keys
        self.n_qpg = n_queries // n_keys
        self.WK = nn.Linear(dim, dim * n_keys)
        self.WV = nn.Linear(dim, dim * n_keys)
        self.WQ = nn.Linear(dim, dim * n_queries)
        self.WO = nn.Linear(dim * n_queries, dim)
        self.is_cosine = is_cosine
        self.norm = nn.LayerNorm(dim) if norm_kvs else nn.Identity()

    def forward_wishful(self, x):
        mask = causal_mask(x)
        seq, bs, dim = x.shape
        K, V, Q = self.WK(x), self.WV(x), self.WQ(x)
        attn = einops.einsum(Q, K, 's1 b (g sub i), s2 b (g i) -> b g sub s1 s2',
                             g=self.n_groups, sub=self.n_qpg, i=dim)
        # Softmax on s2 (kv sequence dim)
        probs = torch.softmax((attn + mask) / dim**.5, dim=4)
        out = einops.einsum(probs, V, 'b g sub s1 s2, s2 b (g i) -> s1 b (g sub i)',
                            g=self.n_groups, sub=self.n_qpg, i=dim)
        return self.WO(out)

    def forward(self, x):
        mask = causal_mask(x)
        seq, bs, dim = x.shape
        K = einops.rearrange(self.WK(x), 's b (g i) -> s b g i', g=self.n_groups, i=dim)
        V = einops.rearrange(self.WV(x), 's b (g i) -> s b g i', g=self.n_groups, i=dim)
        Q = einops.rearrange(self.WQ(x), 's b (g sub i) -> s b g sub i', g=self.n_groups, i=dim)

        # self.norm is identity if not norm_kvs
        K, Q = self.norm(K), self.norm(Q)

        attn = einops.einsum(Q, K, 's1 b g sub i, s2 b g i -> b g sub s1 s2') / dim**.5

        if self.is_cosine:
            probs = attn * torch.exp(mask)
            #probs = probs / torch.linalg.norm(probs, dim=-1, keepdims=True)
            probs = torch.relu(probs)
            probs = probs / probs.sum(dim=-1, keepdims=True)
        else:
            # Softmax on s2 (kv sequence dim)
            probs = torch.softmax((attn + mask), dim=-1)

        out = einops.einsum(probs, V, 'b g sub s1 s2, s2 b g i -> s1 b g sub i')

        out = einops.rearrange(out, 's b g sub i -> s b (g sub i)', g=self.n_groups, sub=self.n_qpg)
        return self.WO(out)

class MyAttention(nn.Module):
    def __init__(self, dim, n_queries, n_keys, crazy_o=False, is_cosine=False, norm_kvs=False, dropout=0, noise_rank=10):
        super().__init__()
        self.n_keys = n_keys
        self.n_queries = n_queries
        self.dropout = dropout
        #Lin = nn.Linear if dropout == 0 else lambda d1, d2: BayesLinear(d1, d2, stdv=dropout)
        #Lin = nn.Linear if dropout == 0 else lambda d1, d2: BayesLinearBias(d1, d2, stdv=dropout)
        Lin = lambda d1, d2: BayesLora(d1, d2, rank=noise_rank, stdv=dropout)
        self.WK = Lin(dim, dim * n_keys)
        self.WV = Lin(dim, dim * n_keys)
        self.WQ = Lin(dim, dim * n_queries)
        self.is_cosine = is_cosine
        self.norm = nn.LayerNorm(dim) if norm_kvs else nn.Identity()
        self.crazy_o = crazy_o
        if crazy_o:
            self.WO = Lin(dim, dim * n_queries)
        else:
            self.WO = Lin(dim * n_queries, dim)

    def forward(self, x):
        mask = causal_mask(x).T
        x = x.transpose(0, 1)  # Put bs first
        bs, seq, dim = x.shape
        K = self.WK(x).reshape(bs, seq, self.n_keys, dim)
        V = self.WV(x).reshape(bs, seq, self.n_keys, dim)
        Q = self.WQ(x).reshape(bs, seq, self.n_queries, dim)

        # self.norm is identity if not norm_kvs
        K, Q = self.norm(K), self.norm(Q)

        # When multiplying a layer with another layer, we have to make sure
        # The normalization is correct. When using a weight layer, we normally
        # try to keep the norm per vector fixed at 1, but when combining two
        # sets of neurons, we expect to get around sqrt(dim), so we divide by that.

        if self.is_cosine:
            attn = torch.einsum('bshd,btid->bhits', Q, K) / dim**.5
            probs = attn * torch.exp(mask)
            probs = probs.flatten(2, 3)
            probs = torch.relu(probs)
            # probs = attn / torch.linalg.norm(attn, dim=2, keepdims=True)
            probs = probs / probs.sum(dim=2, keepdims=True)
        else:
            attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout, is_causal=True)

            # TODO: Could just use torch's fast attention:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # efficient attention using Flash Attention CUDA kernels
            # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

            attn += mask
            # compbine key dim with seq, getting (bs, h, seq2, seq1)
            attn = attn.flatten(2, 3)
            probs = torch.softmax(attn, dim=2)

        # Multiply V on the seq2 dimension (bs, seq, h, dim) -> (bs, h * seq, dim)
        V = V.transpose(1, 2).flatten(1, 2)
        out = torch.einsum('bhts,btd->bhsd', probs, V)
        # Transform to bs(hd) and project hd -> d
        if self.crazy_o:
            O = self.WQ(x).reshape(bs, seq, self.n_queries, dim)
            out = torch.einsum('bhsd,bshd->bsd', out, O)
        else:
            out = self.WO(out.transpose(1,2).flatten(2,3))
        # Back to (seq, bs, dim)
        return out.transpose(0, 1)

class MyTransformerBlock(nn.Module):
    def __init__(self, dim, norm_first, grouped, dropout, noise_rank, **kwargs):
        super().__init__()
        self.norm_first = norm_first
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        if grouped:
            self.attn = GroupedAttention(dim=dim, dropout=dropout, **kwargs)
        else:
            self.attn = MyAttention(dim=dim, dropout=dropout, **kwargs)
        self.ff = nn.Sequential(
            #BayesLinear(dim, 2*dim, stdv=dropout),
            #BayesLinearBias(dim, 2*dim, stdv=dropout),
            BayesLora(dim, 2*dim, rank=noise_rank, stdv=dropout),
            nn.ReLU(),
            #BayesLinear(2*dim, dim, stdv=dropout)
            #BayesLinearBias(2*dim, dim, stdv=dropout)
            BayesLora(2*dim, dim, rank=noise_rank, stdv=dropout)
        )

    def forward(self, x):
        if self.norm_first:
            x = x + self.attn(self.norm0(x))
            x = x + self.ff(self.norm1(x))
        else:
            x = self.norm0(x + self.attn(x))
            x = self.norm1(x + self.ff(x))
        return x


class MyTransformer(nn.Module):
    def __init__(self, n_layers, dim, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[MyTransformerBlock(dim=dim, **kwargs) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(self.norm(self.blocks(x)))


class RevTransformerBlock(nn.Module):
    def __init__(self, dim, dropout, norm_first=True, grouped=False, **kwargs):
        super().__init__()
        assert norm_first and not grouped
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MyAttention(dim=dim, dropout=dropout, **kwargs)
        self.ff = nn.Sequential(
            BayesLinearBias(dim, 2*dim, stdv=dropout),
            nn.ReLU(),
            BayesLinearBias(2*dim, dim, stdv=dropout)
        )

    def forward(self, xy):
        x, y = xy
        y = y + self.attn(self.norm0(x))
        x = x + self.ff(self.norm1(y))
        return x, y


class RevTransformer(nn.Module):
    def __init__(self, n_layers, dim, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[RevTransformerBlock(dim=dim, **kwargs) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x, _ = self.blocks((x, torch.zeros_like(x)))
        return self.fc(self.norm(x))


class ResBlock(nn.Module):
    def __init__(self, dim, groups):
        super().__init__()
        # padding=1 would suffice for keeping the size constant, however, we have
        # to shift everything by one to be causal. For this reason we pad by 2
        # and then throw away the two last generated tokens during forward.
        self.conv1 = nn.Conv1d(dim, 2*dim, kernel_size=3, padding=2, groups=groups)
        self.conv2 = nn.Conv1d(2*dim, dim, kernel_size=3, padding=2, groups=groups)
        # Maybe we don't need this alt path?
        self.alt1 = nn.Conv1d(dim, 6*dim, kernel_size=1, groups=groups)
        self.alt2 = nn.Conv1d(6*dim, dim, kernel_size=1, groups=groups)

    def forward(self, x):
        # Assumes (bs, dim, seq) format
        x = x + self.conv2(torch.relu(self.conv1(x)[:, :, :-2]))[:, :, :-2]
        x = x + self.alt2(torch.relu(self.alt1(x)))
        return x


class ResNet(nn.Module):
    def __init__(self, n_layers, dim, groups):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(dim=dim, groups=groups) for _ in range(n_layers)])

    def forward(self, x):
        # x.shape = (seq, bs, dim)
        x = x.permute(1, 2, 0)  # Resblock expects (bs, dim, seq)
        x = self.blocks(x)
        return x.permute(2, 0, 1)



class TriangularLinear(nn.Linear):
    def __init__(self, n):
        super().__init__(n, n, bias=False)
        with torch.no_grad():
            #torch.nn.init.uniform_(self.weight, -1, 1)
            #indices = torch.arange(n)
            #differences = torch.abs(indices[:, None] - indices[None, :])
            #array = 1 / torch.sqrt(differences + 1) / n**.5
            #self.weight.copy_(torch.tril(self.weight * array))
            # Just copy from the linear layer
            self.weight.copy_(torch.tril(self.weight))
        self.weight.register_hook(lambda grad: grad * torch.tril(torch.ones_like(grad)))


class CausalBlock(nn.Module):
    def __init__(self, seq, dim, rank=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.causals = nn.ModuleList([TriangularLinear(seq) for _ in range(rank)])
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(rank)])

    def forward(self, x):
        # Trying to follow https://arxiv.org/pdf/1603.05027.pdf on the residual design
        inner = torch.relu(self.norm(x))
        total = 0
        for causal, linear in zip(self.causals, self.linears):
            # (seq, batch, dim) -> (batch, dim, seq)
            y = inner.permute(1, 2, 0)
            y = causal(y)
            # (batch, dim, seq) -> (seq, batch, dim)
            y = y.permute(2, 0, 1)
            y = linear(y)
            total += y
        # It looks like x is just the result of an identity transform, but it is not
        # quite so, since the y's all work on inner (which is normalized) rather than
        # x, which is not.
        return total + x


class CausalMLP(nn.Module):
    def __init__(self, num_layers, seq, dim, rank=1):
        super().__init__()
        self.blocks = nn.Sequential(*[CausalBlock(seq, dim, rank) for _ in range(num_layers)])
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(torch.relu(self.blocks(x)))

