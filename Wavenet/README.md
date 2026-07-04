# WaveNet Method: Hierarchical MLP for SMILES Generation

This folder contains a WaveNet-inspired autoregressive SMILES generator.

## File

```text
Wavenet.py
```

## How to Run

From the repository root:

```bash
python Wavenet/Wavenet.py
```

The script runs training, generation, RDKit validity checking, and uniqueness comparison from top to bottom.

Before running on a new machine, update the `pd.read_csv(...)` path inside the script so it points to your local `tg_raw.csv`.

## What This Method Does

The script trains a hierarchical neural language model over SMILES tokens. The workflow is:

1. Load SMILES strings from `tg_raw.csv`.
2. Tokenize SMILES using a hand-defined chemical token vocabulary.
3. Apply BPE-style pair merging to build a vocabulary of size `200`.
4. Add start/end token `0` around each tokenized SMILES.
5. Create next-token examples from a fixed context window.
6. Train a WaveNet-like model using consecutive-token flattening.
7. Sample new SMILES autoregressively.
8. Validate generated strings with RDKit.
9. Compare generated molecules to the training set using canonical SMILES.

## Tokenization

The script starts from the same base SMILES vocabulary as the other methods and then applies pair merges:

```text
vocab_size = 200
num_merges = vocab_size - 52
```

Each merge replaces the most frequent adjacent token pair with a new token. The merged vocabulary is stored in `stoi` and `itos`.

## Model

Main settings:

```text
block_size = 8
batch_size = 32
n_embd = 24
n_hidden = 128
max_iters = 70
learning_rate = 0.0005
generation_cycle = 20000
```

The model uses custom lightweight layers:

- `Embedding`
- `FlattenConsecutive`
- `Linear`
- `BatchNorm1d`
- `Tanh`
- `Sequential`

Active architecture:

```text
Embedding(vocab_size, n_embd)
FlattenConsecutive(2) -> Linear -> BatchNorm1d -> Tanh
FlattenConsecutive(2) -> Linear -> BatchNorm1d -> Tanh
FlattenConsecutive(2) -> Linear -> BatchNorm1d -> Tanh
Linear -> vocab logits
```

This progressively compresses adjacent token groups, similar in spirit to WaveNet-style local context aggregation.

## Training

The script trains with:

- AdamW optimizer
- cross-entropy next-token loss
- train/dev/test split of 80/10/10
- fixed random seed for reproducible shuffling

After training, the script sets each custom layer to evaluation mode so batch normalization uses stored running statistics.

## Generation

Generation starts from an all-zero context:

```text
context = [0] * block_size
```

The model samples one token at a time from the softmax distribution. Sampling stops when token `0` is produced after at least one non-zero token. Generated strings are decoded and filtered through RDKit validity checks.

## Evaluation

The script reports:

- test loss
- valid SMILES percentage
- overlap with the training set
- unique generated SMILES count
- unique generated SMILES percentage

## Notes

- This is a notebook-style experiment script.
- The script currently reads `tg_raw.csv` from an absolute local path. Update the path if running on another machine.
- The model-saving line is commented out.
- The generated-SMILES save-to-file section is commented out.
