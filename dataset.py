import torch
from torch.distributions.geometric import Geometric


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

# Writes [10, 2] ->
# tensor([[0, 1, 0],
#         [0, 0, 2]])
# That is, numbers are right aligned


def numbers_to_digits(tensor, base, max_length):
    """Convert a tensor of numbers to their digit representations."""
    tensor = tensor.unsqueeze(1).repeat(1, max_length)
    bases = torch.pow(
        base, torch.arange(max_length - 1, -1, -1, device=tensor.device)
    ).unsqueeze(0)
    digits = (tensor // bases) % base
    return digits


def digits_to_numbers(digits, base):
    """Convert a tensor of digit representations to their number values."""
    max_length = digits.size(1)
    bases = torch.pow(
        base, torch.arange(max_length - 1, -1, -1, device=digits.device)
    ).unsqueeze(0)
    numbers = (digits * bases).sum(dim=1)
    return numbers


def make_digits_random_length(bs, base, max_number_length):
    digits = torch.randint(base, (bs, max_number_length))
    # n_digits = torch.randint(max_number_length, (bs,))
    n_digits = Geometric(.5).sample((bs,)).clip(max=max_number_length-1)
    mask = torch.arange(max_number_length)[None].repeat(bs, 1) < n_digits[:, None]
    digits[mask] = 0
    return digits


def move_padding_to_end(tensor, padding_token):
    """Move all padding tokens in each row to the end without reordering the rest."""

    # Create a mask for non-padding values
    non_padding_mask = tensor != padding_token

    # Create a tensor with large values where there's padding and row-wise indices elsewhere
    sorting_tensor = non_padding_mask * torch.arange(
        tensor.size(1), device=tensor.device
    ).expand_as(tensor) + (~non_padding_mask) * tensor.size(1)

    # Get the indices that would sort the tensor
    _, sorted_indices = sorting_tensor.sort(dim=1)

    # Use the sorted indices to rearrange the original tensor
    sorted_tensor = torch.gather(tensor, 1, sorted_indices)

    return sorted_tensor


def leading_zeros_to_padding_(digits, padding_token):
    mask = digits.cumsum(1) == 0
    mask[:, -1] = False
    digits[mask] = padding_token


class AdditionDataset:
    def __init__(
        self, num_samples, base, number_length, op):
        self.num_samples = num_samples
        self.base = base
        self.number_length = number_length
        self.sequence_length = 2
        self.op = op

        self.end_token = base  # After input
        self.separator_token = base + 1  # between inputs
        self.padding_token = base + 2  # Before input and after target
        self.eos_token = base + 3  # After target
        self.dot_token = base + 4  # Decimal separator
        self.n_tokens = base + 5

    @property
    def max_input_length(self):
        return self.sequence_length * (self.number_length + 1)

    @property
    def max_output_length(self):
        # Upper bound on total output number length, not including EOS
        if self.op == 'add':
            max_number = self.sequence_length * self.base ** self.number_length
        elif self.op == 'mult':
            max_number = self.base ** (self.number_length * self.sequence_length)
        elif self.op == 'div100':
            max_number = self.base ** self.number_length * 100
        elif self.op == 'sqdiv':
            max_number = (self.base ** self.number_length - 1) ** 2
        elif self.op == 'divmod':
            return 2 * self.number_length + 1
        return len(int_to_digits(max_number, self.base))

    @property
    def seq(self):
        # Upper bound on total length including inputs, outputs, separator and EOS
        return self.max_input_length + self.max_output_length + 1

    def __len__(self):
        return self.num_samples

    def generate_batch(self, bs):
        base = self.base
        # Make batches of random length numbers
        in_digits0 = make_digits_random_length(bs, base, self.number_length)
        in_digits1 = make_digits_random_length(bs, base, self.number_length)
        # Add them together
        numbers0 = digits_to_numbers(in_digits0, base)
        numbers1 = digits_to_numbers(in_digits1, base)
        if self.op == 'divmod':
            numbers1 = torch.clip(numbers1, min=1)
            res0 = numbers_to_digits(numbers0 // numbers1, base, max_length=self.number_length)
            res1 = numbers_to_digits(numbers0 % numbers1, base, max_length=self.number_length)
            leading_zeros_to_padding_(res0, self.padding_token)
            leading_zeros_to_padding_(res1, self.padding_token)
            out_digits = torch.cat([res0, torch.full((bs, 1), self.dot_token), res1], dim=1)
        else:
            if self.op == 'add':
                res = numbers0 + numbers1
            elif self.op == 'mult':
                res = numbers0 * numbers1
            elif self.op == 'div100':
                numbers1 = torch.clip(numbers1, min=1)
                res = numbers0 * 100 // numbers1
            elif self.op == 'sqdiv':
                numbers1 = torch.clip(numbers1, min=1)
                res = numbers0 ** 2 // numbers1
            out_digits = numbers_to_digits(res, base, max_length=self.max_output_length)
            leading_zeros_to_padding_(out_digits, self.padding_token)

        # Replace leading 0s with padding
        leading_zeros_to_padding_(in_digits0, self.padding_token)
        leading_zeros_to_padding_(in_digits1, self.padding_token)

        res = torch.cat(
            [
                in_digits0,
                torch.full((bs, 1), self.separator_token),
                in_digits1,
                torch.full((bs, 1), self.end_token),
                out_digits,
                torch.full((bs, 1), self.eos_token),
            ],
            dim=1,
        )
        res = move_padding_to_end(res, self.padding_token)
        assert res.shape[1] == self.seq, (res.shape, self.seq)
        return res

    def repr_example(self, example):
        dic = {i: str(i) for i in range(self.base + 1)}
        dic[self.padding_token] = ""
        dic[self.end_token] = " = "
        dic[self.eos_token] = ""
        dic[self.separator_token] = {
                "mult": " * ",
                "add": " + ",
                "div100": " / ",
                "divmod": " / ",
                "sqdiv": "^2 / ",
            }[self.op]
        dic[self.dot_token] = "."
        return "".join(dic[token.item()] for token in example)

