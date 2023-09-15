import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import AdditionDataset
import methods


def sinusoidal_position_embeddings(seq_length, d_model):
    """
    Args:
    - seq_length (int): The length of the sequence for which position embeddings are required.
    - d_model (int): Dimension of the model (embedding size).

    Returns:
    - position_embeddings (Tensor): Sinusoidal position embeddings of shape (seq_length, d_model).
    """

    # Create a matrix of positions
    position = torch.arange(seq_length).unsqueeze(1).float()  # shape: [seq_length, 1]

    # Create a matrix of dimension indices
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * -(torch.log(torch.tensor(10000.0)) / d_model)
    )  # shape: [d_model/2]

    # Compute sinusoidal position embeddings
    position_embeddings = torch.empty(seq_length, d_model)
    position_embeddings[:, 0::2] = torch.sin(position * div_term)
    position_embeddings[:, 1::2] = torch.cos(position * div_term)

    return position_embeddings


class AdditionModel(nn.Module):
    def __init__(
        self,
        kind,
        ds,
        hidden_size,
        num_layers,
        num_heads,
        lr,
        dropout,
    ):
        super().__init__()
        self.ds = ds  # Input the dataset for relevant parameters
        self.lr = lr
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(ds.n_tokens, hidden_size)
        self.kind = kind
        seq = self.ds.seq
        if kind == "lstm":
            self.model = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=False,
            )
        elif kind == "rnn":
            self.model = nn.RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif kind == "gru":
            self.model = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif kind == "transformer-lstm":
            self.base = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    dim_feedforward=hidden_size * 2,
                    nhead=num_heads,
                    norm_first=True,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers - 1,
            )
        elif kind == "attention-rope":
            self.model = nn.Sequential(*[
                    methods.RotaryEmbeddingTransformerLayer(hidden_size, num_heads, 0, dropout)
                for _ in range(num_layers)
            ])
        elif kind == "transformer-rope":
            self.model = nn.Sequential(*[
                    methods.RotaryEmbeddingTransformerLayer(hidden_size, num_heads, hidden_size * 2, dropout)
                for _ in range(num_layers)
            ])
        elif kind == "transformer-alibi":
            self.model = nn.Sequential(*[
                    methods.AlibiTransformerLayer(hidden_size, num_heads, hidden_size * 2, dropout)
                for _ in range(num_layers)
            ])
        elif kind.startswith("transformer"):
            if kind == "transformer":
                self.pos_emb = nn.Embedding(seq, hidden_size)
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    dim_feedforward=hidden_size * 2,
                    nhead=num_heads,
                    norm_first=True,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers,
            )
        elif kind == "hybrid":
            self.model1 = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=(num_layers + 1) // 2,
                dropout=dropout,
                batch_first=True,
            )
            self.model2 = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    dim_feedforward=hidden_size * 2,
                    nhead=num_heads,
                    norm_first=True,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers // 2,
            )
        else:
            raise Error(f"Kind {kind} is not supported")
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x.shape = (batch, seq)
        x = self.embedding(x)
        bs, seq, dim = x.shape

        if self.kind in ("lstm", "rnn", "gru"):
            x, _ = self.model(x)
        elif self.kind.startswith("transformer"):
            if self.kind == "transformer":
                if self.pos_emb.num_embeddings < seq:
                    print(f"Increasing pos embedding size from {self.pos_emb.num_embeddings} to {seq}")
                    with torch.no_grad():
                        new_pos_emb = nn.Embedding(seq, self.pos_emb.embedding_dim).to(x.device)
                        # Copy old positional embeddings
                        new_pos_emb.weight[: self.pos_emb.num_embeddings] = self.pos_emb.weight
                        self.pos_emb = new_pos_emb
                positions = torch.arange(seq).unsqueeze(0).to(x.device)
                emb = self.pos_emb(positions).to(x.device)
                x = x + emb
            elif self.kind == "transformer-sine":
                emb = sinusoidal_position_embeddings(seq, dim).to(x.device)
                x = x + emb
            elif self.kind == "transformer-lstm":
                x, _ = self.base(x)
            # My own transformers don't need a causal mask
            if self.kind in ("transformer-rope", "transformer-alibi"):
                x = self.model(x)
            else:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(seq, x.device)
                x = self.model(x, mask=attn_mask, is_causal=True)
        elif self.kind == "hybrid":
            x, _ = self.model1(x)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(seq, x.device)
            x = self.model2(x, attn_mask, is_causal=True)
        else:
            x = self.model(x)
        return self.fc(self.norm(x))

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 1e-2},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if our data is on cuda
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, fused=self.embedding.weight.is_cuda
        )
        return optimizer

    @torch.no_grad()
    def generate(self, input_sequence):
        # Generate output sequence from input sequence using trained model
        # Future featuers:
        #  - Beam search: We use greedy digit-by-digit generation. It's possible the models
        #    would perform a bit better if we instead used beam search to pick the most
        #    likely overall result.
        #  - Support generating on multiple sequences at the same time. Would presumably
        #    require some padding or keeping track of the length of each existing sequence.
        #  - KV caching: Instead of evaluating the model on the whole sequence for every
        #    token generated, we can save cache the KVs of the previous tokens and only
        #    compute one column of the model at a time.
        assert input_sequence[-1] == self.ds.end_token, "Input should end with ="
        # Pad to expected length
        n = len(input_sequence)
        input_sequence = torch.cat(
            [
                input_sequence,
                torch.full((self.ds.seq - n,), self.ds.padding_token, device=input_sequence.device),
            ]
        )
        with torch.no_grad():
            for i in range(n, self.ds.seq):
                output_logits = self(input_sequence[None])[0, i - 1]
                token = torch.argmax(output_logits)
                input_sequence[i] = token
        return input_sequence[n:]

    @torch.no_grad()
    def print_examples(self, num_examples=3, must_include_a_wrong=False):
        examples = self.ds.generate_batch(num_examples).to(self.embedding.weight.device)
        i = 0
        while i < num_examples:
            example = examples[i]
            # Cut answer off example, and generate an answer using the model instead:
            n = example.tolist().index(self.ds.end_token) + 1
            true_answer = example[n:]
            raw_prediction = self.generate(example[:n])
            is_correct = torch.all(true_answer == raw_prediction)
            # Get at least one wrong example each time
            if is_correct and i == num_examples - 1 and must_include_a_wrong:
                examples[i] = self.ds.generate_batch(1)[0]
                continue
            print("Example:", self.ds.repr_example(example))
            print("Output: ", self.ds.repr_example(raw_prediction), f"({'Correct' if is_correct else 'Wrong'})")
            print("Raw In: ", example.tolist())
            print("Raw Out:", raw_prediction.tolist())
            i += 1
