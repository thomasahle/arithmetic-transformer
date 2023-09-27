import subprocess
import re
import sys
from tqdm import tqdm
from tabulate import tabulate


def run_program_with_parameters(num_layers, num_heads):
    # Run the program with the given parameters
    cmd = [
        sys.executable, "train.py",
        "--base", "2",
        "--train-batches", "100",
        "--val-batches", "10",
        "--batch-size", "1000",
        "--kind", "transformer-nope",
        "--epochs", "2",
        "--flip",
        "--hidden-size", "60",
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # Extract the largest number of digits completed
    matches = re.findall(r'Switching to number length (\d+)', output)
    if matches:
        # Return the largest number found
        return int(matches[-1])
    else:
        return None


def print_table(headers, rows):
    # Clear previous table display
    for _ in range(len(rows) + 2):  # +2 accounts for headers and initial table display line
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.write("\033[K")  # Clear current line
    # Print updated table
    print(tabulate(rows, headers=headers, tablefmt='grid'))


def main():
    num_layers_values = list(range(1, 7))  # example values, adjust as needed
    num_heads_values = list(range(1, 7))  # example values, adjust as needed
    
    best_value = 0
    best_parameters = None

    # 2D list to store results
    results_table = [["-" for _ in num_heads_values] for _ in num_layers_values]
    
    headers = ["num-layers/num-heads"] + num_heads_values
    print(tabulate([["-" for _ in headers]], headers=headers, tablefmt='grid'))  # Initial table display

    with tqdm(total=len(num_layers_values) * len(num_heads_values), desc="Progress") as pbar:
        for i, num_layers in enumerate(num_layers_values):
            for j, num_heads in enumerate(num_heads_values):
                result = run_program_with_parameters(num_layers, num_heads)
                if result:
                    results_table[i][j] = result
                    if result > best_value:
                        best_value = result
                        best_parameters = (num_layers, num_heads)
                
                # Print the table with the updated result
                rows = [[num_layers] + row for num_layers, row in zip(num_layers_values, results_table)]
                print_table(headers, rows)
                
                pbar.update(1)

    print(f"\nThe best value is {best_value} with num-layers = {best_parameters[0]} and num-heads = {best_parameters[1]}")

if __name__ == "__main__":
    main()

