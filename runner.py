import subprocess
import re
import sys
from tqdm import tqdm
from tabulate import tabulate
import argparse
import textwrap

def run_program_with_parameters(num_layers, num_heads, args):
    cmd = [
        sys.executable,
        "train.py",
        "--flip",
        "--num-layers",
        str(num_layers),
        "--num-heads",
        str(num_heads),
    ]
    for k, v in vars(args).items():
        if k not in ["max_layers", "max_heads", "command", "outfile"]:
            cmd.extend([f"--{k.replace('_','-')}", str(v)])
    print(cmd)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print('Retrying...')
        return run_program_with_parameters(num_layers, num_heads, args)
    matches = re.findall(r"Switching to number length (\d+)", result.stdout)
    print(result.stderr)
    return int(matches[-1]) - 1 if matches else 0


def print_table(headers, rows):
    # Clear previous table display
    for _ in range(3 * len(rows) + 2):  # +2 accounts for headers and initial table display line
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.write("\033[K")  # Clear current line
    # Print updated table
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def save_to_file(headers, rows, filename):
    with open(filename, "w") as file:
        file.write(",".join(headers) + "\n")
        for row in rows:
            file.write(",".join(map(str, row)) + "\n")


def run(args):
    num_layers_values = list(range(1, args.max_layers + 1))
    num_heads_values = [h for h in range(1, args.max_heads + 1) if args.hidden_size % h == 0]

    best_value = 0
    best_parameters = None

    # 2D list to store results
    results_table = [["-" for _ in num_heads_values] for _ in num_layers_values]

    headers = ["num-layers/num-heads"] + list(map(str, num_heads_values))
    print(tabulate([["-" for _ in headers]], headers=headers, tablefmt="grid"))  # Initial table display

    with tqdm(total=len(num_layers_values) * len(num_heads_values), desc="Progress") as pbar:
        for i, num_layers in enumerate(num_layers_values):
            for j, num_heads in enumerate(num_heads_values):
                result = run_program_with_parameters(num_layers, num_heads, args)
                results_table[i][j] = result
                if result > best_value:
                    best_value = result
                    best_parameters = (num_layers, num_heads)

                # Print the table with the updated result
                rows = [[num_layers] + row for num_layers, row in zip(num_layers_values, results_table)]
                print_table(headers, rows)

                pbar.update(1)

    rows = [[num_layers] + row for num_layers, row in zip(num_layers_values, results_table)]
    save_to_file(headers, rows, args.outfile)

    print(
        f"\nThe best value is {best_value} with num-layers = {best_parameters[0]} and num-heads = {best_parameters[1]}"
    )


def plot(args):
    import numpy as np
    import matplotlib.pyplot as plt

    # Load data and headers from the CSV file
    data = np.genfromtxt(args.filename, delimiter=",", skip_header=1, dtype=int)
    with open(args.filename, "r") as f:
        xheaders = f.readline().strip().split(",")[1:]

    fig, ax = plt.subplots()
    cax = ax.matshow(data[:, 1:], cmap="Greys_r", origin="lower", vmin=0, vmax=64)
    fig.colorbar(cax)

    # Set tick locations and labels
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(len(xheaders)))
    ax.set_xticklabels(xheaders)

    yheaders = data[:, 0]
    ax.set_yticks(np.arange(len(yheaders)))
    ax.set_yticklabels(yheaders)

    ax.set_xlabel("num-heads")
    ax.set_ylabel("num-layers")

    wrapper = textwrap.TextWrapper(width=60)
    ax.set_title(wrapper.fill(args.title))

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="A utility to run train.py with different parameters or to plot results."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run subparser
    run_parser = subparsers.add_parser("run", help="Run train.py with different parameters.")
    run_parser.add_argument("--max-layers", type=int, default=6, help="Values for num-layers parameter.")
    run_parser.add_argument("--max-heads", type=int, default=6, help="Values for num-heads parameter.")
    run_parser.add_argument("--outfile", type=str, help="Where to save the results")
    run_parser.add_argument("--dropout", type=float, default=0.05)
    run_parser.add_argument("--base", type=int, default=2, help="Base argument for train.py.")
    run_parser.add_argument("--epochs", type=int, default=2)
    run_parser.add_argument(
        "--train-batches",
        type=int,
        default=100,
        help="train-batches argument for train.py.",
    )
    run_parser.add_argument("--val-batches", type=int, default=10, help="val-batches argument for train.py.")
    run_parser.add_argument("--batch-size", type=int, default=1000, help="batch-size argument for train.py.")
    run_parser.add_argument("--kind", type=str, default="transformer-nope", help="kind argument for train.py.")
    run_parser.add_argument("--hidden-size", type=int, default=60, help="hidden-size argument for train.py.")

    # Plot subparser
    plot_parser = subparsers.add_parser("plot", help="Visualize results.")
    plot_parser.add_argument("filename", default="results.csv", help="Name of the file to plot data from.")
    plot_parser.add_argument("--title", help="Title of plot")

    args = parser.parse_args()

    if args.command == "run":
        run(args)
    elif args.command == "plot":
        plot(args)


if __name__ == "__main__":
    main()
