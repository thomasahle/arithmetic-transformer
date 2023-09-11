
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import AdditionDataset
import methods

def causal_mask(x):
    """ matrix with -inf above the diagonal and 0 on the diagonal and below """
    sz = len(x)
    mask = torch.ones(sz, sz).triu(1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(x.device)


class AdditionModel(pl.LightningModule):
    def __init__(self, kind, ds, batch_size, hidden_size, num_layers, num_heads, num_queries, lr, norm_first, grouped, is_cosine, norm_kvs, dropout, noise_rank):
        super().__init__()
        self.ds = ds  # Input the dataset for relevant parameters
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_size = hidden_size
        num_tokens = ds.base + 4  # 4 extra tokens for end, separator, padding, and eos
        self.embedding = nn.Embedding(num_tokens, hidden_size)
        self.kind = kind
        seq = self.ds.seq
        if kind == "lstm":
            self.model = nn.LSTM( input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        elif kind == "rnn":
            self.model = nn.RNN( input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        elif kind == "gru":
            self.model = nn.GRU( input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        elif kind == 'myformer':
            self.pos_emb = nn.Embedding(seq, hidden_size)
            self.model = methods.MyTransformer(
                n_layers=num_layers,
                dim=hidden_size,
                n_queries=num_queries,
                n_keys=num_heads,
                norm_first=norm_first,
                grouped=grouped,
                is_cosine=is_cosine,
                norm_kvs=norm_kvs,
                dropout=dropout,
                noise_rank=noise_rank,
            )
        elif kind == 'revformer':
            self.pos_emb = nn.Embedding(seq, hidden_size)
            self.model = methods.RevTransformer(
                n_layers=num_layers,
                dim=hidden_size,
                n_queries=num_queries,
                n_keys=num_heads,
                norm_first=norm_first,
                grouped=grouped,
                is_cosine=is_cosine,
                norm_kvs=norm_kvs,
                dropout=dropout,
                noise_rank=noise_rank,
            )
        elif kind == "transformer":
            self.pos_emb = nn.Embedding(seq, hidden_size)
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size, dim_feedforward=hidden_size * 2, nhead=num_heads,
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
                    d_model=hidden_size, dim_feedforward=hidden_size * 2, nhead=num_heads,
                    norm_first=norm_first,
                    dropout=dropout,
                ),
                num_layers // 2,
            )
        elif kind == "mlp":
            if num_layers == 1:
                print("Warning, MLP doesn't support single layer")
            layers = [methods.TriangularLinear(hidden_size * seq), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.append(methods.TriangularLinear(hidden_size * seq))
                layers.append(nn.ReLU())
            layers.append(methods.TriangularLinear(hidden_size * seq))
            self.model = nn.Sequential(*layers)
        elif kind == "mlp2":
            self.model = methods.CausalMLP(num_layers, seq, hidden_size, rank=num_heads)
        elif kind == "res":
            self.model = methods.ResNet(n_layers=num_layers, dim=hidden_size, groups=num_heads)
        else:
            raise Error(f"Kind {kind} is not supported")

    def forward(self, x):
        # x.shape = (batch, seq)
        emb = self.embedding(x)
        # emb.shape = (batch, seq, dim)
        # Note lstm assumes (seq, batch, feature)
        x = emb.permute(1, 0, 2)

        if self.pos_emb.num_embeddings < x.size(0):
            print(f"Increasing pos embedding size from {self.pos_emb.num_embeddings} to {x.size(0)}")
            with torch.no_grad():
                new_pos_emb = nn.Embedding(x.size(0), self.pos_emb.embedding_dim).to(x.device)
                new_pos_emb.weight[:self.pos_emb.num_embeddings] = self.pos_emb.weight
                self.pos_emb = new_pos_emb

        if self.kind in ("lstm", "rnn", "gru"):
            x, _ = self.model(x)
        elif self.kind in ("myformer", "revformer"):
            positions = torch.arange(0, x.size(0)).unsqueeze(0).to(x.device)
            emb = self.pos_emb(positions).permute(1, 0, 2).to(x.device)
            x = self.model(x + emb)
        elif self.kind == "transformer":
            positions = torch.arange(0, x.size(0)).unsqueeze(0).to(x.device)
            emb = self.pos_emb(positions).permute(1, 0, 2).to(x.device)
            #x = self.model(x + emb, torch.zeros_like(x), tgt_mask=causal_mask(x))
            attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device)
            x = self.model(x + emb, mask=attn_mask, is_causal=True)
            # Add an extra linear layer, just becuase I'm doing this in myformer.
            x = self.fc(self.norm(x))
        elif self.kind == "hybrid":
            x, _ = self.model1(x)
            #x = self.model2(x, torch.zeros_like(x), tgt_mask=causal_mask(x))
            attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device)
            x = self.model2(x, attn_mask, is_causal=True)
        elif self.kind == "mlp":
            x = self.model(emb.flatten(start_dim=1)).reshape(emb.shape)
            x = x.permute(1, 0, 2)
        elif self.kind in ("mlp2", "res"):
            x = self.model(x)
        # Re-permute to (batch, seq, dim)
        x = x.permute(1, 0, 2)
        # Might as well reuse the embeddings
        return x @ self.embedding.weight.T

    def training_step(self, batch, batch_idx):
        # Mask of everything from end_token and to the right
        mask = torch.cumsum(batch == self.ds.end_token, dim=1)[:, 1:]
        truth = batch[:, 1:][mask == 1]
        out = self(batch)[:, :-1][mask == 1]

        # Compute loss only on the targets, not the input
        loss = nn.functional.cross_entropy(out, truth)

        # Bayesian loss
        bayes_loss = 0
        for module in self.modules():
            if hasattr(module, 'penalty'):
                bayes_loss += module.penalty()
        # 1e-5 gives a very janky learning. 1e-3 is often too much and doesn't
        # allow the model to learn.
        loss += 1e-4 * bayes_loss

        return loss

    def validation_step(self, batch, batch_idx):
        # We validate only past the end-token
        mask = torch.cumsum(batch == self.ds.end_token, dim=1)[:, 1:]
        # We shift everything by one, since we are autoregressive
        truth = batch[:, 1:][mask == 1]
        out = self(batch)[:, :-1]
        logits = out[mask == 1]

        loss = nn.functional.cross_entropy(logits, truth)
        self.log("val_loss", loss, prog_bar=True)

        wn = 0
        for pn, p in self.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                wn += (p ** 2).mean()
        self.log("weigth_norm", wn, prog_bar=True)

        # bl = 0
        # for module in self.modules():
        #     if hasattr(module, 'penalty'):
        #         bl += module.penalty()
        # self.log("bayes_loss", bl, prog_bar=True)

        # Count only test cases completely solved.
        # Otherwise we get too many points for predicting padding.
        truth2 = batch[:, 1:] * mask
        preds = torch.argmax(out, dim=2) * mask
        acc = torch.all(preds == truth2, dim=1).float().mean()
        self.log("val_acc", acc, prog_bar=True)

        # if self.kind == 'mlp2' and batch_idx == 0 and hasattr(self.logger.experiment, 'log'):
        #     w2sum = 0
        #     for i, block in enumerate(self.model.blocks):
        #         for j, causal in enumerate(block.causals):
        #             w2 = causal.weight**2
        #             w2sum += w2
        #             wandb_img = wandb.Image(cm.hot(w2.cpu().numpy()), caption="Causal Norms")
        #             self.logger.experiment.log({f"causal norms {i} head {j}": wandb_img}, step=self.global_step)

        #     w2sum /= len(self.model.blocks)
        #     wandb_img = wandb.Image(cm.hot(w2sum.cpu().numpy()), caption="Causal Norms")
        #     self.logger.experiment.log({"causal norms": wandb_img}, step=self.global_step)

        return acc

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 1e-2},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, fused=torch.cuda.is_available())
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        # Return a DataLoader for the newly created dataset.
        # The idea is to make it bigger as we learn
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        # Make a smaller dataset for validation
        print(f"Making validation dataloader with number_length={self.ds.number_length}")
        dataset = AdditionDataset(
            self.ds.num_samples // 10,
            base=self.ds.base,
            number_length=self.ds.number_length,
            min_sequence_length=self.ds.sequence_length,
            sequence_length=self.ds.sequence_length,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=1)

    def on_validation_epoch_end(self):
        # Get the logged accuracy for the current epoch
        current_val_acc = self.trainer.callback_metrics.get('val_acc', 0)  # Assuming 'val_acc' is the key you use to log accuracy

        print(f"\nGot validation accuracy {current_val_acc}.")
        self.print_examples()

        if current_val_acc > 0.9:
            # Switch to a more difficult dataset
            print(f"Switching to number length {self.ds.number_length + 1}!")
            self.ds = AdditionDataset(
                self.ds.num_samples,
                base=self.ds.base,
                # Increase number length
                number_length=self.ds.number_length + 1,
                min_sequence_length=self.ds.sequence_length,
                sequence_length=self.ds.sequence_length,
            )

    def generate(self, input_sequence):
        # Generate output sequence from input sequence using trained model
        assert input_sequence[-1] == self.ds.end_token
        # Pad to expected length
        n = len(input_sequence)
        input_sequence = torch.cat([
            torch.tensor(input_sequence),
            torch.tensor([self.ds.padding_token] * (self.ds.seq - n), dtype=int)
        ]).to(self.device)
        with torch.no_grad():
            for i in range(n, self.ds.seq):
                output_logits = self(input_sequence[None])[0, i - 1]
                token = torch.argmax(output_logits)
                input_sequence[i] = token
        return input_sequence[n:]

    def print_examples(self, num_examples=3):
        with torch.no_grad():
            dic = {i: str(i) for i in range(self.ds.base+1)}
            dic[self.ds.padding_token] = ''
            dic[self.ds.end_token] = ','
            dic[self.ds.eos_token] = ''
            dic[self.ds.separator_token] = ','
            for example in self.ds.generate_batch(num_examples):
                example = list(example.cpu().numpy())
                print(example)
                # [list(group) for k, group in itertools.groupby(l, lambda x: x >= 2) if not k]
                string = ''.join(dic[t] for t in example)
                nums = [num for num in string.split(',')]
                print("Example:", ' + '.join(nums[:-1]), '=', nums[-1])

                n = example.index(self.ds.end_token) + 1
                prediction = self.generate(example[:n]).cpu().numpy()
                string = ''.join(dic[t] for t in prediction)
                print("Output: ", string, prediction)
                print()

