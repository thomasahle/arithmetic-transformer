# Learning Arithmetic with Sequence Models
This repository contains code for training neural networks to learn the process of addition.
We experiment with different causal neural network architectures like LSTM, Transformer, and various positional encodings. The best one being a transformer *using a one layer LSTM as layer 1*.
The primary focus of this project is to compare how quickly different models learn to add numbers with varying digit lengths.

## Results
Here is a summary of the number of epochs different models took to learn addition with various digit lengths:

### 4 layers, 32 hidden size, 1 head
The table shows the number of epochs needed to learn addition of `n` digits.
E.g. the Hybrid model took just 7 epochs (7*10^6 examples) to learn 8 digit addition to 90% accuracy, after it had already learned 7 digit addition to 90% accuracy.

To learn 14 digit addition, the hybrid model took a total of 135 epochs ~ 10^8 examples.
This is of course much less than the total of different 10^28 possible input pairs.
The larger models (below) are able to learn even faster.

|Digits| Transformer Learned | Transformer Sine | Transformer NoPE | Transformer LSTM | LSTM | Hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 2 | 3 | 1 | 2 | 2 |
| 3 | 2 | 3 | 4 | 2 | 7 | 3 |
| 4 | 21 | 6 | 8 | 3 | 11 | 5 |
| 5 | 16 | 5 | 15 | 7 | 24 | 8 |
| 6 | 81 | 9 | 33 | 4 | 59 | 9 |
| 7 | 270+ | 7 | 82 | 3 | 52 | 8 |
| 8 | - | 15 | 282 | 9 | - | 7 |
| 9 | - | 27 | - | 8 | - | 11 |
| 10 | - | 42 | - | 10 | - | 20 |
| 11 | - | 44 | - | 12 | - | 33 |
| 12 | - | - | - | 21 | - | 34 |
| 13 | - | - | - | 27 | - | 48 |
| 14 | - | - | - | 27 | - | 103 |

Each model in the table had roughly 32K parameters.


### 4 layers, 64 hidden size, 4 heads

Increasing the model sizes just slightly allowed the model to learn multi-digit addition much faster.
The hybrid model got up to 18 digits with just a few epochs per digit, before something finally broke during the 19 digit training and accuracy fell to near 0.

|Digits| T. Learned | T. Sine | T. NoPE | T. LSTM | LSTM | Hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 1 | 1 | 1 | 2 | 1 |
| 3 | 1 | 1 | 2 | 1 | 3 | 2 |
| 4 | 2 | 2 | 2 | 1 | 5 | 2 |
| 5 | 2 | 1 | 2 | 1 | 10 | 3 |
| 6 | 3 | 1 | 1 | 1 | 15 | 5 |
| 7 | 4 | 1 | 2 | 1 | 11 | 1 |
| 8 | 10 | 3 | 2 | 1 | 12 | 1 |
| 9 | 6 | 1 | 3 | 1 | 18 | 2 |
| 10 | 10 | 1 | 4 | 1 | 15 | 1 |
| 11 | 32 | 2 | 1 | 1 | 18 | 1 |
| 12 | 27 | 2 | 4 | 1 | 21 | 2 |
| 13 | 137 | 2 | 9 | 1 | 30 | 2 |
| 14 | 105 | 2 | 13 | 2 | - | 1 |
| 15 | 123 | 2 | 20 | 1 | - | 1 |
| 16 | 162+ | 2 | - | 1 | - | 6 |
| 17 | - | 3 | - | 1 | - | 3 |
| 18 | - | 2 | - | 1 | - | 5 |


 
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

### Positional encoding

Since the input size can in principle grow without limit, I had to decide how to do positional encoding in the transformers.
I tried not using positional encoding at all (NoPE), and using an nn.Embedding where I simply added new vectors as needed.
Not having embeddings was a bit better than the additive positional embeddings, but the best results came from using an LSTM for the first two layers, and then switch to a transformer for the top 2. I called this the "hybrid" method.

## More Results

### Multiplication

I also trained some 800K parameter models to learn multiplication.
Each one is 6 layers, 4 hears, 128 hidden size and 1% dropout.

| Digits | T. Learned | T. Sine | T. NoPE | T. LSTM | LSTM  | Hybrid |
|:------:|:----------:|:-------:|:-------:|:-------:|:-----:|:------:|
|   1    |     1      |    1    |    1    |    1    |   1   |   1    |
|   2    |     2      |    2    |    2    |    1    |   5   |   1    |
|   3    |    15      |   16    |   18    |   14    |  41   |  34    |
|   4    |   181      |  233    |   67    |  133    | 953+  | 803+   |
|   5    |   507+     |  426+   |  411    |  349+   |   -   |   -    |
|   6    |     -      |    -    |  122+   |    -    |   -   |   -    |

Surprisingly the position encoding less transformer does best here, though it's possible the Transformer with LSTM on the first layer would have done as well, if I had let it run as long.

### Division, DivMod

| Digits | T. LSTM | T. Sine |
|:------:|:-------:|:-------:|
|   1    |    1    |    1    |
|   2    |    1    |    1    |
|   3    |    3    |    3    |
|   4    |    8    |   10    |
|   5    |   22    |   45    |
|   6    |   53    |   74    |
|   7    |  123    |  156    |
|   8    |   21+    |    2+    |

Example outputs from training:```
Example: 715564 / 22242 = 32.3820
Output:  32.3820 (Correct)
Example: 8711 / 816706 = 0.8711
Output:  0.8711 (Correct)
Example: 35173 / 6 = 5862.1
Output:  5862.1 (Correct)
```

### Division, SqDiv

| Digits | T. LSTM | T. Sine |
|:------:|:-------:|:-------:|
|   1    |    1    |    1    |
|   2    |    2    |    2    |
|   3    |  179    |  281    |
|   4    |   23+    |  101+    |

Example outputs from training:
```
Example: 282^2 / 1640 = 48   
Output:  48 (Correct)        
Example: 7^2 / 112 = 0       
Output:  0 (Correct)         
Example: 8287^2 / 920 = 74646
Output:  74660 (Wrong)       
```
