import argparse
import itertools
import torch
import tqdm
from collections import Counter
import torch.nn.functional as F

from dataset import AdditionDataset
from model import AdditionModel


def main():
    # Needed to enable tensor cores
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of examples to generate and train on",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam LR")
    parser.add_argument(
        "--acc-next", type=float, default=0.9, help="Accuracy before next level"
    )
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="The hidden size for the neural network",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="The number of layers for the neural network",
    )
    parser.add_argument("--batch-size", type=int, default=2**10, help="Batch size")
    parser.add_argument(
        "--kind",
        type=str,
        help="The type of neural network to use (lstm, transformer, hybrid)",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="add",
        help="Operation to learn (add, mult)",
    )
    parser.add_argument("--norm-last", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--num-heads",
        type=int,
        default=1,
        help="The number of heads/rank in transformer/mlp",
    )
    args = parser.parse_args()

    dataset = AdditionDataset(
        10**6,  # data points per epoch
        base=10,
        number_length=1,
        op=args.op
    )

    model = AdditionModel(
        ds=dataset,
        kind=args.kind,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        norm_first=not args.norm_last,
        lr=args.lr,
        dropout=args.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters")

    # model = torch.compile(model)
    manual_training(model, dataset, args)


def manual_training(model, dataset, args):
    if args.cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model = model.to(device)

    # Get optimizer (and potentially the scheduler)
    optimizers = model.configure_optimizers()
    if isinstance(optimizers, tuple) and len(optimizers) == 2:
        optimizer, scheduler = optimizers
    else:
        optimizer = optimizers

    # Standard PyTorch Training Loop
    time_to_success = Counter()
    for epoch in range(args.epochs):
        train_batches = 1000
        with torch.no_grad():
            X = dataset.generate_batch(args.batch_size * train_batches).to(device)
            Xmask = (torch.cumsum(X == dataset.end_token, dim=1) == 1) & (
                X != dataset.end_token
            )
            Xmask = Xmask[:, 1:]

        # Training Loop
        model.train()
        for batch_idx in tqdm.tqdm(range(train_batches)):
            batch = X[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            mask = Xmask[
                batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
            ]

            optimizer.zero_grad()
            truth = batch[:, 1:]
            out = model(batch)[:, :-1]
            loss = F.cross_entropy(out[mask], truth[mask])
            loss.backward()
            optimizer.step()

        model.print_examples(3)

        # Validation Loop
        accs = []
        model.eval()
        with torch.no_grad():
            val_batches = 100
            X = dataset.generate_batch(args.batch_size * val_batches).to(device)
            Xmask = (torch.cumsum(X == dataset.end_token, dim=1) == 1) & (
                X != dataset.end_token
            )
            Xmask = Xmask[:, 1:]

            for batch_idx in tqdm.tqdm(range(val_batches)):
                batch = X[
                    batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
                ]
                mask = Xmask[
                    batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
                ]

                truth2 = batch[:, 1:] * mask
                out = model(batch)[:, :-1]
                preds = torch.argmax(out, dim=2) * mask
                acc = torch.all(preds == truth2, dim=1).float().mean()
                accs.append(acc)

        time_to_success[dataset.number_length] += 1

        acc = torch.mean(torch.tensor(accs))
        print(f"Validation acc: {acc}")
        print(sorted(time_to_success.items()))
        if acc > args.acc_next:
            print(f"Switching to number length {dataset.number_length+1}")
            print(f"Took {time_to_success[dataset.number_length]} epochs")
            dataset.number_length += 1


if __name__ == "__main__":
    main()
