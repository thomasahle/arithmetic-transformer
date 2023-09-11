import torch
from torch.utils.data import Dataset, DataLoader, random_split

def int_to_digits(n, base):
    """Convert an integer to a list of digits in a given base."""
    digits = []
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits or [0]


def digits_to_int(digits, base):
    """Convert a list of digits in a given base to an integer."""
    n = 0
    for digit in list(digits)[::-1]:
        n = n * base + digit
    return n


def format_size(n):
    if n >= 1e9: return f'{n / 1e9:.4}G'
    elif n >= 1e6: return f'{n / 1e6:.4}M'
    elif n >= 1e3: return f'{n / 1e3:.4}K'
    else: return str(n)


class AdditionDataset(Dataset):
    def __init__(
        self, num_samples, base, number_length, sequence_length, min_sequence_length=1
    ):
        self.num_samples = num_samples
        self.base = base
        self.number_length = number_length
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length

        self.end_token = base  # After input
        self.separator_token = base + 1  # between inputs
        self.padding_token = base + 2  # Before input and after target
        self.eos_token = base + 3  # After target

    @property
    def max_input_length(self):
        return self.sequence_length * (self.number_length + 1)

    @property
    def max_output_length(self):
        # Upper bound on total output length, including EOS token
        return len(int_to_digits(self.sequence_length * self.base**self.number_length, self.base)) + 1

    @property
    def seq(self):
        # Upper bound on total length including inputs, outputs and separators
        return self.max_input_length + self.max_output_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = []
        total = 0

        # for _ in range(torch.randint(self.min_sequence_length, self.sequence_length+1, (1,))):
        for _ in range(self.sequence_length):
            if self.min_sequence_length == 1:
               n_digits = torch.randint(1, self.number_length+1, (1,))
            else:
                n_digits = self.number_length
            number_digits = torch.randint(self.base, (n_digits,))
            sequence += list(number_digits) + [self.separator_token]
            total += digits_to_int(number_digits, self.base)

        # End of sequence should be END, not SEP
        sequence[-1] = self.end_token
        sequence += int_to_digits(total, self.base) + [self.eos_token]

        # Distribute padding left and right
        n_padding = self.seq - len(sequence)
        left_padding = torch.randint(0, n_padding+1, (1,))
        sequence = [self.padding_token] * left_padding + sequence + [self.padding_token] * (n_padding - left_padding)

        return torch.tensor(sequence)

    def generate_batch(self, bs):
        base = self.base
        # Make batches of random length numbers
        in_digits0 = make_digits_random_length(bs, base, self.number_length)
        in_digits1 = make_digits_random_length(bs, base, self.number_length)
        # Add them together
        sums = digits_to_numbers(in_digits0, base) + digits_to_numbers(in_digits1, base)
        out_digits = numbers_to_digits(sums, base, max_length = self.number_length+1)
        # Replace leading 0s with padding
        leading_zeros_to_padding_(out_digits, self.padding_token)
        leading_zeros_to_padding_(in_digits0, self.padding_token)
        leading_zeros_to_padding_(in_digits1, self.padding_token)

        res = torch.cat([
            in_digits0,
            torch.full((bs, 1), self.separator_token),
            in_digits1,
            torch.full((bs, 1), self.end_token),
            out_digits,
            torch.full((bs, 1), self.eos_token)
        ], dim=1)
        res = move_padding_to_end(res, self.padding_token)
        return res



# Writes [10, 2] ->
# tensor([[0, 1, 0],
#         [0, 0, 2]])
# That is, numbers are right aligned

def numbers_to_digits(tensor, base, max_length):
    """Convert a tensor of numbers to their digit representations."""
    tensor = tensor.unsqueeze(1).repeat(1, max_length)
    bases = torch.pow(base, torch.arange(max_length-1, -1, -1, device=tensor.device)).unsqueeze(0)
    digits = (tensor // bases) % base
    return digits

def digits_to_numbers(digits, base):
    """Convert a tensor of digit representations to their number values."""
    max_length = digits.size(1)
    bases = torch.pow(base, torch.arange(max_length-1, -1, -1, device=digits.device)).unsqueeze(0)
    numbers = (digits * bases).sum(dim=1)
    return numbers

def make_digits_random_length(bs, base, max_number_length):
    digits = torch.randint(base, (bs, max_number_length))
    n_digits = torch.randint(max_number_length, (bs,))
    mask = torch.arange(max_number_length)[None].repeat(bs, 1) < n_digits[:, None]
    digits[mask] = 0
    return digits

def move_padding_to_end(tensor, padding_token):
    """Move all padding tokens in each row to the end without reordering the rest."""
    
    # Create a mask for non-padding values
    non_padding_mask = tensor != padding_token

    # Create a tensor with large values where there's padding and row-wise indices elsewhere
    sorting_tensor = non_padding_mask * torch.arange(tensor.size(1), device=tensor.device).expand_as(tensor) \
                     + (~non_padding_mask) * tensor.size(1)

    # Get the indices that would sort the tensor
    _, sorted_indices = sorting_tensor.sort(dim=1)

    # Use the sorted indices to rearrange the original tensor
    sorted_tensor = torch.gather(tensor, 1, sorted_indices)

    return sorted_tensor

def leading_zeros_to_padding_(digits, padding_token):
    mask = (digits.cumsum(1) == 0)
    mask[:, -1] = False
    digits[mask] = padding_token

