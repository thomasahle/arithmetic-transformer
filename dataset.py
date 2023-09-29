import numpy as np
import math
from random import randrange, choice
import itertools


# Adding extra padding is an easy way to improve performance, as it gives the
# model more space to think. For example, without padding, the standard model
# kind=transformer-lstm gets accuracies (1, 0.95, 0.66), but we just five extra
# paddings on the left, it gets (1, 0.98, 0.80). Even better, if we add the
# padding right before the equality sign, ...

class Dataset:
    def __init__(self, base, number_length, pre_end_padding=0,
                 flip=False, preferred_dtype='int64'):
        self.base = base
        self.number_length = number_length
        self.pre_end_padding = pre_end_padding
        self.flip = flip
        self.preferred_dtype = preferred_dtype

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
        if np.dtype(self.preferred_dtype) is np.dtype(object):
            powers = [self.base**(i+1) for i in range(number_length)]
            n = math.prod(shape)
            result = np.array([randrange(choice(powers)) for i in range(n)],
                              dtype=object).reshape(shape)
        else:
            digits_shape = shape + (number_length,)
            digits = np.random.randint(0, self.base, math.prod(digits_shape)).reshape(digits_shape)
            n_digits = np.random.randint(0, number_length, shape)
            mask = np.arange(number_length) < n_digits[..., None]
            exponents = np.arange(number_length - 1, -1, -1, dtype=self.preferred_dtype)
            bases = np.expand_dims(np.power(self.base, exponents), 0)
            result = (digits * bases).sum(axis=-1)
        return result

    def to_digits(self, numbers, length=None):
        if length is None:
            length = self.number_length

        # Convert numbers to digits
        tensor = np.tile(np.expand_dims(numbers, 1), (1, length))
        exponents = np.arange(length - 1, -1, -1, dtype=self.preferred_dtype)
        bases = np.expand_dims(np.power(self.base, exponents), 0)
        digits = (tensor // bases) % self.base

        # Mask leading zeros
        mask = digits.cumsum(1) == 0
        mask[:, -1] = False
        digits[mask] = self.padding_token
        if self.flip:
            return np.flip(digits, [1])
        return digits

    def move_padding_to_end(self, tensor, end=True):
        """Move all padding tokens in each row to the end without reordering the rest."""

        # Create a tensor with large values where there's padding and row-wise indices elsewhere
        # This allows us to "sort" the padding to the end, while keeping everything else in its
        # original order.
        sorting_tensor = np.where(
            tensor == self.padding_token,
            tensor.shape[1] if end else -tensor.shape[1],
            np.arange(tensor.shape[1])
        )

        # Get the indices that would sort the tensor
        sorted_indices = np.argsort(sorting_tensor, axis=1)

        # Use the sorted indices to rearrange the original tensor
        sorted_tensor = np.take_along_axis(tensor, sorted_indices, 1)

        return sorted_tensor

    def generate_batch(self, bs):
        # TODO: Could insert COT padding here
        res = self._generate_batch(bs)
        res = self.move_padding_to_end(res)
        res[res == self.n_tokens] = self.padding_token
        assert res.shape == (bs, self.seq)
        return res.astype(int)

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
    def __init__(self, base, number_length, pre_end_padding=0, flip=False, **kwargs):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)
        self.separator_token2 = self.n_tokens
        self.n_tokens += 1
        self.dic[self.separator_token] = "+"
        self.dic[self.separator_token2] = "%"

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs))
        c = self.make_numbers((bs,), (self.number_length + 1) // 2)
        out = (a + b) % np.clip(c, 1, None)
        return np.concatenate(
            [
                np.full((bs, 1), self.start_token),
                self.to_digits(a),
                np.full((bs, 1), self.separator_token),
                self.to_digits(b),
                np.full((bs, 1), self.separator_token2),
                self.to_digits(c),
                np.full((bs, 1), self.end_token),
                self.to_digits(out),
                np.full((bs, 1), self.eos_token),
            ],
            axis=1,
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
        **kwargs,
    ):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)
        self.func = func
        self.sep_string = sep
        self.out_length = out_length
        self.dic[self.separator_token] = sep
        self.min_b = min_b

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs))
        b = np.clip(b, self.min_b, None)
        out = self.func(a, b)
        return np.concatenate(
            [
                np.full((bs, 1), self.start_token),
                self.to_digits(a),
                np.full((bs, 1), self.separator_token),
                self.to_digits(b),
                np.full((bs, 1), self.end_token),
                self.to_digits(out, length=self.out_length),
                np.full((bs, 1), self.eos_token),
            ],
            axis=1,
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
        **kwargs,
    ):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)
        self.output_separator = self.n_tokens
        self.n_tokens += 1
        self.dic[self.separator_token] = "/%"
        self.dic[self.output_separator] = ","

    def _generate_batch(self, bs):
        a, b = self.make_numbers((2, bs), self.base)
        b = np.clip(b, 1, None)
        div = a // b
        mod = a % b
        return np.concatenate(
            [
                np.full((bs, 1), self.start_token),
                self.to_digits(a),
                np.full((bs, 1), self.separator_token),
                self.to_digits(b),
                np.full((bs, 1), self.end_token),
                self.to_digits(div),
                np.full((bs, 1), self.output_separator),
                self.to_digits(mod),
                np.full((bs, 1), self.eos_token),
            ],
            axis=1,
        )

    @property
    def seq(self):
        return self.number_length * 4 + 5


class FactorDataset(Dataset):
    def __init__(self, base, number_length, pre_end_padding=0, flip=False, **kwargs):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)
        self.dic[self.separator_token] = "*"
        self.primes = None
        self.primes_length = 0

    def get_primes(self, number_length):
        if self.primes_length == number_length:
            return self.primes
        n = self.base**self.number_length
        sieve = np.ones(n, dtype=bool)
        # We include 1, but not 0
        sieve[0] = False
        for i in range(2, n):
            if sieve[i]:
                sieve[i * i :: i] = False
        self.primes = np.squeeze(np.nonzero(sieve))
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
        # A random number contains the factor p with probability 1/p
        weights = 1 / primes
        num_samples = bs * self.max_factors
        sampled_primes = random.choices(primes, weights=weights, k=num_samples)
        sampled_primes = np.array(sampled_primes, dtype=self.preferred_dtype)
        sampled_primes = np.reshape(sampled_primes, (bs, self.max_factors))
        # Products may be too large. Let's fix that
        while True:
            log_prods = np.sum(np.log(sampled_primes.astype(float)), axis=1)
            mask = log_prods > math.log(self.base) * self.number_length
            num_too_large = mask.sum().item()
            if num_too_large == 0:
                break
            # Update the first non-one value in the rows that are too large
            non_one_indices = (sampled_primes != 1).astype(int)
            first_non_one_mask = (
                np.cumsum(non_one_indices, axis=1) == 1
            ) & np.expand_dims(mask, -1)
            sampled_primes[first_non_one_mask] = 1

        filtered_primes = sampled_primes
        prods = np.prod(filtered_primes, axis=1)
        filtered_primes = np.sort(filtered_primes, axis=1)
        parts = [
            np.full((bs, 1), self.start_token),
            self.to_digits(prods),
            np.full((bs, 1), self.end_token),
        ]
        for i in range(self.max_factors):
            parts += [
                self.to_digits(filtered_primes[:, i]),
                np.full((bs, 1), self.separator_token),
            ]
            # If 1, change it to padding
            parts[-2][filtered_primes[:, i] == 1] = self.padding_token
            parts[-1][filtered_primes[:, i] == 1] = self.padding_token
        # Replace last separator with EOS
        parts[-1] = np.full((bs, 1), self.eos_token)
        res = np.concatenate(parts, axis=1)
        res = self.move_padding_to_end(res)
        res = res[:, : self.seq]
        return res
