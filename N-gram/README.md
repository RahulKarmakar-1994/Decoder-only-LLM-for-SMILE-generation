# N-gram Method: Feed-forward SMILES Language Model

This folder contains a feed-forward n-gram neural language model for SMILES generation.

## File

```text
n-gram_w_batchnorm_block_size_varry.py
```

## How to Run

From the repository root:

```bash
python N-gram/n-gram_w_batchnorm_block_size_varry.py
```

The script runs training, generation, RDKit validity checking, and uniqueness comparison from top to bottom.

Before running on a new machine, update the `pd.read_csv(...)` path inside the script so it points to your local `tg_raw.csv`.

## What This Method Does

The script trains a neural next-token predictor using a fixed-length context window. The workflow is:

1. Load SMILES strings from `tg_raw.csv`.
2. Tokenize SMILES using a hand-defined chemical token vocabulary.
3. Build token sequences with start/end token `0`.
4. Create next-token examples from a fixed context window.
5. Train an embedding plus single-hidden-layer MLP.
6. Sample new SMILES autoregressively.
7. Validate generated strings with RDKit.
8. Compare canonical generated SMILES with canonical training SMILES.

## Tokenization

The method starts from a 52-token SMILES vocabulary:

```text
vocab_size = 52
```

The BPE merge section is present, but `num_merges = vocab_size - 52`, so no additional BPE merges are applied in the current configuration.

## Model

Main settings:

```text
block_size = 12
batch_size = 32
n_embd = 24
n_hidden = 200
max_iters = 100
learning_rate = 0.01
generation_cycle = 20000
```

The model parameters are manually defined:

```text
C   token embedding matrix
W1  input-to-hidden weight
b1  hidden bias
W2  hidden-to-output weight
b2  output bias
```

Forward pass:

```text
context tokens -> embeddings -> flatten -> tanh hidden layer -> logits
```

Training uses cross-entropy loss and manual gradient updates.

## Generation

Generation starts with an all-zero context:

```text
context = [0] * block_size
```

The model samples one token at a time from the softmax distribution. Generation stops when token `0` is sampled after at least one non-zero token. Generated token sequences are decoded back into SMILES strings and validated with RDKit.

## Evaluation

The script reports:

- training loss
- validation loss
- test loss
- valid SMILES percentage
- number of overlapping generated/training SMILES
- number of unique generated SMILES
- unique SMILES percentage

## Notes

- This is a notebook-style experiment script.
- The script currently reads `tg_raw.csv` from an absolute local path. Update the path if running on another machine.
- The save-to-file section is commented out.
- The file name says `batchnorm`, but the active model in this script is the manually parameterized MLP without the custom batch-normalized layer stack used in the WaveNet script.
