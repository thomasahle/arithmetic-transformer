# Learning Arithmetic with Sequence Models
This repository contains code for training neural networks to learn the process of addition.
We experiment with different causal neural network architectures like LSTM, Transformer, and a hybrid model which is *half LSTM, half Transformer*.
The primary focus of this project is to compare how quickly different models learn to add numbers with varying digit lengths.

## Results
Here is a summary of the number of epochs different models took to learn addition with various digit lengths:

### 4 layers, 32 hidden size
The table shows the number of epochs needed to learn addition of `n` digits.
E.g. the Hybrid model took just 7 epochs (7*10^6 examples) to learn 8 digit addition to 90% accuracy, after it had already learned 7 digit addition to 90% accuracy.

To learn 14 digit addition, the hybrid model took a total of 293 epochs ~ 3*10^8 examples.
This is of course much less than the total of different 10^28 possible input pairs.

|Digits| Transformer | Transformer NoPE | LSTM | Hybrid |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 3 | 2 | 2 |
| 3 | 2 | 4 | 7 | 3 |
| 4 | 21 | 8 | 11 | 5 |
| 5 | 16 | 15 | 24 | 8 |
| 6 | 81 | 33 | 59 | 9 |
| 7 | - | 82 | 52 | 8 |
| 8 | - | - | - | 7 |
| 9 | - | - | - | 11 |
| 10 | - | - | - | 20 |
| 11 | - | - | - | 33 |
| 12 | - | - | - | 34 |
| 13 | - | - | - | 48 |
| 14 | - | - | - | 103 |

Each model in the table had roughly 32K parameters.

### 4 layers, 64 hidden size
|Digits| Transformer | Transformer NoPE | LSTM | Hybrid |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 2 | 2 | 1 |
| 3 | 2 | 2 | 3 | 2 |
| 4 | 3 | 2 | 5 | 2 |
| 5 | 3 | 5 | 10 | 5 |
| 6 | 4 | 9 | 15 | 5 |
| 7 | 33 | 18 | 11 | 5 |
| 8 | - | - | 12 | 6 |
| 9 | - | - | 18 | 7 |
| 10 | - | - | 15 | 3 |
| 11 | - | - | 18 | 4 |
| 12 | - | - | - | 2 |
| 13 | - | - | - | 2 |
| 14 | - | - | - | 2 |
| 15 | - | - | - | 5 |
| 16 | - | - | - | 1 |


Roughly 130K parameters per model.

## Installation
Clone the repository

```bash
git clone https://github.com/thomasahle/arithmetic-transformer
cd arithmetic-transformer
pip install -r requirements.txt
```

## Example

```
$ py train.py --kind lstm --dropout .01
The model has 34240 parameters
100%|███████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:10<00:00, 99.51it/s]
Data:    [9, 11, 8, 10, 1, 7, 13]
Example: 9 + 8 = 17
Output:  17 - Raw: [ 1  7 13]
Data:    [3, 11, 3, 10, 6, 13, 12]
Example: 3 + 3 = 6
Output:  6 - Raw: [ 6 13 12]
Data:    [3, 11, 1, 10, 4, 13, 12]
Example: 3 + 1 = 4
Output:  4 - Raw: [ 4 13 12]
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 752.11it/s]
Validation acc: 0.8921972513198853
...
Validation acc: 0.9629980325698853
[(1, 2)]
Switching to number length 2
100%|███████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:13<00:00, 74.09it/s]
Data:    [5, 11, 6, 10, 1, 1, 13, 12, 12, 12]
Example: 5 + 6 = 11
Output:  1 - Raw: [ 1 13 12 12 12 12]
Data:    [2, 11, 1, 7, 10, 1, 9, 13, 12, 12]
Example: 2 + 17 = 19
Output:  1 - Raw: [ 1 13 12 12 12]
Data:    [2, 9, 11, 3, 10, 3, 2, 13, 12, 12]
Example: 29 + 3 = 32
Output:  45 - Raw: [ 4  5 13 12 12]
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 752.11it/s]
Validation acc: 0.05396484211087227
...
```

In this example we train a 34K parameter LSTM.
After 2 epochs it has a 96% accuracy on 1 digit addition, so we switch to 2 digits, and the accuracy falls to 5%.
Once the model has achieved 90%+ accuracy on 2 digit addition, we switch to 3 digits and so on.

## Details

Each model was given example input in the format
```
[d, d, d, ..., SEP, e, e, ..., END, f, f, ..., EOS, PAD, ...]
```
Where `d, ...` and `e, ...` are the numbers to be added, and `f, ...` is the sum.
SEP, END, EOS and PAD are extra tokens, representing "+", "=" and extra padding to align examples of different lengths.
All numbers are in most significant digit first order.
Doing least significant digit first makes the problem easier, but less realistic for real world text.
