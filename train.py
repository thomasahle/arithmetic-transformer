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
    parser.add_argument("--dropout", type=float, default=0.05)
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
        required=True,
        type=str,
        help="The type of neural network to use (lstm, transformer, hybrid)",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="add",
        help="Operation to learn (add, mult)",
    )
    parser.add_argument(
        "--cot-padding",
        type=int,
        default=0,
        help="Chain of thought padding",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=10,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--flip", action="store_true", help="Flip order of numbers")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="The number of heads/rank in transformer/mlp",
    )
    args = parser.parse_args()

    dataset = AdditionDataset(
        base=args.base,
        number_length=1,
        op=args.op,
        pre_end_padding=args.cot_padding,
        flip=args.flip,
    )

    model = AdditionModel(
        ds=dataset,
        kind=args.kind,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        dropout=args.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters")

    if args.compile:
        model = torch.compile(model)
    manual_training(model, dataset, args)


def answer_mask(dataset, batch):
    """Creates a mask of everything after the END (or =) token, which separates the question
    from the answer."""
    mask = (torch.cumsum(batch == dataset.end_token, dim=1) == 1)
    mask &= (batch != dataset.end_token)
    return mask[:, 1:]


def training_step(model, batch):
    """Computes cross entropy loss between the model output and the ground truth, but only on
    the tokens after the END token, since the previous data is just random."""
    mask = answer_mask(model.ds, batch)
    truth = batch[:, 1:]
    out = model(batch)[:, :-1]
    return F.cross_entropy(out[mask], truth[mask])


def validation_step(model, batch):
    """Computes the accuracy on the model, if we assume greedy decoding is used.
    We only consider a question corectly solved if every single token is correctly predicted,
    including the padding."""
    mask = answer_mask(model.ds, batch)
    truth = batch[:, 1:]
    out = model(batch)[:, :-1]
    preds = torch.argmax(out, dim=2)

    # We'd to test that our validation method matches what you get with generate.
    # Unfortunately the LSTMs give slightly different results when passing a batch,
    # vs when passing one element at a time, which breaks the direct correspondance.
    for i in range(0):
        n = batch[i].tolist().index(model.ds.end_token) + 1
        true = batch[i, n:]
        pred0 = preds[i, n-1:]
        pred1 = model.generate(batch[i][:n])
        if torch.all((preds * mask)[i] == (truth * mask)[i]):
            assert torch.all(pred0 == true)
            # If we are getting the answer right, they should be the same.
            assert torch.all(pred0 == pred1)
        else:
            # If we are getting the answer wrong, they should both be wrong.
            assert not torch.all(pred0 == true)
            assert not torch.all(pred1 == true)

    return torch.all(preds * mask == truth * mask, dim=1).float().mean()


def manual_training(model, dataset, args):
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    batch_size = args.batch_size
    optimizer = model.configure_optimizers()

    # Standard PyTorch Training Loop
    time_to_success = Counter()
    for epoch in range(args.epochs):
        train_batches = 1000
        with torch.no_grad():
            train_data = dataset.generate_batch(batch_size * train_batches).to(device)

        # Training Loop
        model.train()
        for batch_idx in tqdm.tqdm(range(train_batches)):
            batch = train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            optimizer.zero_grad()
            loss = training_step(model, batch)
            loss.backward()
            optimizer.step()

        # Validation Loop
        accs = []
        model.eval()
        with torch.no_grad():
            val_batches = 100
            val_data = dataset.generate_batch(batch_size * val_batches).to(device)

            for batch_idx in tqdm.tqdm(range(val_batches)):
                batch = val_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                acc = validation_step(model, batch)
                accs.append(acc)
        acc = torch.mean(torch.tensor(accs))
        print(f"Validation acc: {acc:.5}")

        # Print some examples. Try to always include an example where the model is wrong.
        # But if the model is nearly perfect, don't bother, since we might search forever.
        model.print_examples(3, must_include_a_wrong=acc<args.acc_next)

        time_to_success[dataset.number_length] += 1

        print("Epochs per digit:", sorted(time_to_success.items()))
        if acc > args.acc_next:
            print(f"Switching to number length {dataset.number_length+1}")
            dataset.number_length += 1


if __name__ == "__main__":
    main()
