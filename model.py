import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import AdditionDataset


class AdditionModel(nn.Module):
    def __init__(
        self,
        kind,
        ds,
        hidden_size,
        num_layers,
        num_heads,
        lr,
        norm_first,
        dropout,
    ):
        super().__init__()
        self.ds = ds  # Input the dataset for relevant parameters
        self.lr = lr
        self.hidden_size = hidden_size
        num_tokens = ds.base + 4  # 4 extra tokens for end, separator, padding, and eos
        self.embedding = nn.Embedding(num_tokens, hidden_size)
        self.kind = kind
        seq = self.ds.seq
        if kind == "lstm":
            self.model = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif kind == "rnn":
            self.model = nn.RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif kind == "gru":
            self.model = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif kind in ("transformer", "transformer-nope"):
            self.pos_emb = nn.Embedding(seq, hidden_size)
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    dim_feedforward=hidden_size * 2,
                    nhead=num_heads,
                    norm_first=norm_first,
                    dropout=dropout,
                ),
                num_layers,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Linear(hidden_size, hidden_size)
        elif kind == "hybrid":
            self.model1 = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=(num_layers + 1) // 2,
                dropout=dropout,
            )
            self.model2 = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    dim_feedforward=hidden_size * 2,
                    nhead=num_heads,
                    norm_first=norm_first,
                    dropout=dropout,
                ),
                num_layers // 2,
            )
        else:
            raise Error(f"Kind {kind} is not supported")

    def forward(self, x):
        # x.shape = (batch, seq)
        emb = self.embedding(x)
        # emb.shape = (batch, seq, dim)
        # Note lstm assumes (seq, batch, feature)
        x = emb.permute(1, 0, 2)

        if hasattr(self, "pos_emb") and self.pos_emb.num_embeddings < x.size(0):
            print(
                f"Increasing pos embedding size from {self.pos_emb.num_embeddings} to {x.size(0)}"
            )
            with torch.no_grad():
                new_pos_emb = nn.Embedding(x.size(0), self.pos_emb.embedding_dim).to(
                    x.device
                )
                new_pos_emb.weight[: self.pos_emb.num_embeddings] = self.pos_emb.weight
                self.pos_emb = new_pos_emb

        if self.kind in ("lstm", "rnn", "gru"):
            x, _ = self.model(x)
        elif self.kind in ("transformer", "transformer-nope"):
            positions = torch.arange(0, x.size(0)).unsqueeze(0).to(x.device)
            if self.kind != "transformer-nope":
                emb = self.pos_emb(positions).permute(1, 0, 2).to(x.device)
                x = x + emb
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                x.shape[1], x.device
            )
            x = self.model(x, mask=attn_mask, is_causal=True)
            x = self.fc(self.norm(x))
        elif self.kind == "hybrid":
            x, _ = self.model1(x)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                x.shape[1], x.device
            )
            x = self.model2(x, attn_mask, is_causal=True)
        # Re-permute to (batch, seq, dim)
        x = x.permute(1, 0, 2)
        # Might as well reuse the embeddings
        return x @ self.embedding.weight.T

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
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, fused=torch.cuda.is_available()
        )
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def generate(self, input_sequence):
        # Generate output sequence from input sequence using trained model
        assert input_sequence[-1] == self.ds.end_token
        # Pad to expected length
        n = len(input_sequence)
        input_sequence = torch.cat(
            [
                torch.tensor(input_sequence),
                torch.tensor([self.ds.padding_token] * (self.ds.seq - n), dtype=int),
            ]
        ).to(self.embedding.weight.device)
        with torch.no_grad():
            for i in range(n, self.ds.seq):
                output_logits = self(input_sequence[None])[0, i - 1]
                token = torch.argmax(output_logits)
                input_sequence[i] = token
        return input_sequence[n:]

    def print_examples(self, num_examples=3):
        with torch.no_grad():
            dic = {i: str(i) for i in range(self.ds.base + 1)}
            dic[self.ds.padding_token] = ""
            dic[self.ds.end_token] = ","
            dic[self.ds.eos_token] = ""
            dic[self.ds.separator_token] = ","
            for example in self.ds.generate_batch(num_examples):
                example = list(example.cpu().numpy())
                print("Data:   ", example)
                string = "".join(dic[t] for t in example)
                nums = [num for num in string.split(",")]
                print("Example:", " + ".join(nums[:-1]), "=", nums[-1])

                n = example.index(self.ds.end_token) + 1
                prediction = self.generate(example[:n]).cpu().numpy()
                string = "".join(dic[t] for t in prediction)
                print("Output: ", string, "- Raw:", prediction)
