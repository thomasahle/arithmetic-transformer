import torch
import math
import itertools


# Adding extra padding is an easy way to improve performance, as it gives the
# model more space to think. For example, without padding, the standard model
# kind=transformer-lstm gets accuracies (1, 0.95, 0.66), but we just five extra
# paddings on the left, it gets (1, 0.98, 0.80). Even better, if we add the
# padding right before the equality sign, ...


class Dataset:
    def __init__(self, base, number_length, pre_end_padding=0, flip=False):
        self.base = base
        self.number_length = number_length
        self.pre_end_padding = pre_end_padding
        self.flip = flip

        self.start_token = base  # Before input
        self.end_token = base + 1  # After input
        self.separator_token = base + 2  # between inputs
        self.padding_token = base + 3  # Before input and after target
        self.eos_token = base + 4  # After target
        self.n_tokens = base + 5

        self.dic = {i: str(i) for i in range(self.base + 1)}
        self.dic[self.padding_token] = ""
        self.dic[self.start_token] = ""
        self.dic[self.end_token] = "="
        self.dic[self.eos_token] = ""

    def make_numbers(self, shape, number_length=None):
        if number_length is None:
            number_length = self.number_length
        digits = torch.randint(self.base, shape + (number_length,))
        n_digits = torch.randint(number_length, shape)
        mask = torch.arange(number_length) < n_digits[..., None]
        digits[mask] = 0
        bases = torch.pow(self.base, torch.arange(number_length - 1, -1, -1)).unsqueeze(
            0
        )
        return (digits * bases).sum(dim=-1)

    def to_digits(self, numbers, length=None):
        if length is None:
            length = self.number_length

        # Convert numbers to digits
        tensor = numbers.unsqueeze(1).repeat(1, length)
        bases = torch.pow(
            self.base, torch.arange(length - 1, -1, -1, device=tensor.device)
        ).unsqueeze(0)
        digits = (tensor // bases) % self.base

        # Mask leading zeros
        mask = digits.cumsum(1) == 0
        mask[:, -1] = False
        digits[mask] = self.padding_token
        if self.flip:
            return torch.flip(digits, [1])
        return digits

    def move_padding_to_end(self, tensor, end=True):
        """Move all padding tokens in each row to the end without reordering the rest."""

        # Create a tensor with large values where there's padding and row-wise indices elsewhere
        # This allows us to "sort" the padding to the end, while keeping everything else in its
        # original order.
        sorting_tensor = torch.where(
            tensor == self.padding_token,
            tensor.size(1) if end else -tensor.size(1),
            torch.arange(tensor.size(1), device=tensor.device),
        )

        # Get the indices that would sort the tensor
        _, sorted_indices = sorting_tensor.sort(dim=1)

        # Use the sorted indices to rearrange the original tensor
        sorted_tensor = torch.gather(tensor, 1, sorted_indices)

        return sorted_tensor

    def generate_batch(self, bs):
        # TODO: Could insert COT padding here
        res = self._generate_batch(bs)
        res = self.move_padding_to_end(res)
        res[res == self.n_tokens] = self.padding_token
        assert res.shape == (bs, self.seq)
        return res

    def _generate_batch(self, tokens):
        assert False, "Not implemented"

    def repr_example(self, example):
        tokens = [
            (tuple(group)[::-1] if self.flip else tuple(group))
            if is_number
            else next(group)
            for is_number, group in itertools.groupby(
                example.tolist(), key=lambda x: x < self.base
            )
        ]
        return self._repr_tokens(tokens).strip()

    def _repr_tokens(self, tokens):
        res = []
        for token in tokens:
            if type(token) is tuple:
                res.append("".join(map(str, token)))
            else:
                res.append(self.dic[token])
        return " ".join(res)

    @property
    def seq(self):
        assert False, "Not implemented"


class AddModDataset(Dataset):
    def __init__(self, base, number_length, pre_end_padding=0, flip=False):
        super().__init__(base, number_length, pre_end_padding, flip)
        self.separator_token2 = self.n_tokens
        self.n_tokens += 1
        self.dic[self.separator_token] = "+"
        self.dic[self.separator_token2] = "%"

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs))
        c = self.make_numbers((bs,), (self.number_length + 1) // 2)
        out = (a + b) % torch.clip(c, min=1)
        return torch.cat(
            [
                torch.full((bs, 1), self.start_token),
                self.to_digits(a),
                torch.full((bs, 1), self.separator_token),
                self.to_digits(b),
                torch.full((bs, 1), self.separator_token2),
                self.to_digits(c),
                torch.full((bs, 1), self.end_token),
                self.to_digits(out),
                torch.full((bs, 1), self.eos_token),
            ],
            dim=1,
        )

    @property
    def seq(self):
        return self.number_length * 4 + 5


class BinaryOpDataset(Dataset):
    def __init__(
        self,
        base,
        number_length,
        func,
        sep,
        out_length,
        pre_end_padding=0,
        min_b=0,
        flip=False,
    ):
        super().__init__(base, number_length, pre_end_padding, flip)
        self.func = func
        self.sep_string = sep
        self.out_length = out_length
        self.dic[self.separator_token] = sep
        self.min_b = min_b

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs))
        b = torch.clip(b, min=self.min_b)
        out = self.func(a, b)
        return torch.cat(
            [
                torch.full((bs, 1), self.start_token),
                self.to_digits(a),
                torch.full((bs, 1), self.separator_token),
                self.to_digits(b),
                torch.full((bs, 1), self.end_token),
                self.to_digits(out, length=self.out_length),
                torch.full((bs, 1), self.eos_token),
            ],
            dim=1,
        )

    @property
    def seq(self):
        return self.number_length * 2 + self.out_length + 4


class DivModDataset(Dataset):
    def __init__(
        self,
        base,
        number_length,
        pre_end_padding=0,
        flip=False,
    ):
        super().__init__(base, number_length, pre_end_padding, flip)
        self.output_separator = self.n_tokens
        self.n_tokens += 1
        self.dic[self.separator_token] = "/%"
        self.dic[self.output_separator] = ","

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs), self.base)
        b = torch.clip(b, min=1)
        div = a // b
        mod = a % b
        return torch.cat(
            [
                torch.full((bs, 1), self.start_token),
                self.to_digits(a),
                torch.full((bs, 1), self.separator_token),
                self.to_digits(b),
                torch.full((bs, 1), self.end_token),
                self.to_digits(div),
                torch.full((bs, 1), self.output_separator),
                self.to_digits(mod),
                torch.full((bs, 1), self.eos_token),
            ],
            dim=1,
        )

    @property
    def seq(self):
        return self.number_length * 4 + 5


class FactorDataset(Dataset):
    def __init__(self, base, number_length, pre_end_padding=0, flip=False):
        super().__init__(base, number_length, pre_end_padding, flip)
        self.dic[self.separator_token] = ","
        self.primes = None
        self.primes_length = 0

    def get_primes(self, number_length):
        if self.primes_length == number_length:
            return self.primes
        n = self.base ** self.number_length
        sieve = torch.ones(n, dtype=torch.bool)
        # We include 1, but not 0
        sieve[0] = False
        for i in range(2, n):
            if sieve[i]:
                sieve[i*i::i] = False
        self.primes = torch.nonzero(sieve).squeeze()
        self.primes_length = self.number_length
        return self.primes

    @property
    def max_factors(self):
        return int(math.log2(self.base) * self.number_length)

    @property
    def seq(self):
        # task is "length", answer is longest if all factors are 2.
        # finally 3 extra tokens: start, end and eos.
        # actually one less, because we need one separator less than factors
        return self.number_length + 2 * self.max_factors + 2

    def _generate_batch(self, bs):
        primes = self.get_primes(self.number_length)
        weights = 1/primes
        # Using rejection sampling to
        for c in itertools.count(2):
            indices = torch.multinomial(weights, num_samples=2**c * bs * self.max_factors, replacement=True)
            sampled_primes = primes[indices].reshape(2**c * bs, self.max_factors)
            log_prods = torch.sum(torch.log(sampled_primes), dim=1)
            mask = log_prods < math.log(self.base) * self.number_length
            if mask.sum() < bs:
                print(f'Notice: Got {mask.sum()} primes, wanted {bs}.')
                continue
            filtered_primes = sampled_primes[mask][:bs]
            break
        prods = torch.prod(filtered_primes, dim=1)
        filtered_primes = filtered_primes.sort(dim=1).values
        parts = [
            torch.full((bs, 1), self.start_token),
            self.to_digits(prods),
            torch.full((bs, 1), self.end_token),
        ]
        for i in range(self.max_factors):
            parts += [
                self.to_digits(filtered_primes[:, i]),
                torch.full((bs, 1), self.separator_token),
            ]
            # If 1, change it to padding
            parts[-2][filtered_primes[:, i] == 1] = self.padding_token
            parts[-1][filtered_primes[:, i] == 1] = self.padding_token
        # Replace last separator with EOS
        parts[-1] = torch.full((bs, 1), self.eos_token)
        res = torch.cat(parts, dim=1)
        res = self.move_padding_to_end(res)
        res = res[:, :self.seq]
        return res

