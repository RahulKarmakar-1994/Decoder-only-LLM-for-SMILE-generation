# LLM Method: Decoder-only Transformer for SMILES Generation

This folder contains the Transformer-based SMILES generator and a separate script for generation from a saved checkpoint.

## Quick Start

Run these commands from the repository root:

```bash
cd LLM
```

Train the Transformer and save the checkpoint:

```bash
python LLM.py
```

Generate SMILES from the saved checkpoint:

```bash
python generate_from_checkpoint.py \
  --checkpoint byte_pair_attention_model_vocab_200.pth \
  --num-smiles 1000
```

The generated SMILES are saved to:

```text
generated_smiles_from_ckpt.txt
```

To generate more molecules, change `--num-smiles`:

```bash
python generate_from_checkpoint.py \
  --checkpoint byte_pair_attention_model_vocab_200.pth \
  --num-smiles 10000
```

## Files

```text
LLM.py
generate_from_checkpoint.py
byte_pair_attention_model_vocab_200.pth
tg_raw.csv
README.md
```

## What This Method Does

`LLM.py` trains a decoder-only Transformer language model over SMILES tokens. The workflow is:

1. Load SMILES strings from `tg_raw.csv`.
2. Tokenize SMILES using a hand-defined chemical token vocabulary.
3. Build a BPE-style merged vocabulary with target vocabulary size `200`.
4. Add start/end token `0` around each sequence.
5. Build next-token training examples with context length `block_size = 8`.
6. Train a causal self-attention model.
7. Save a checkpoint dictionary containing model weights, vocabulary, BPE merges, and model hyperparameters.

## Model

The model is named `BigramLanguageModel` in the code, but it is a decoder-only Transformer rather than a simple bigram model.

Main settings in `LLM.py`:

```text
vocab_size = 200
block_size = 8
batch_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
max_iters = 70
learning_rate = 0.0005
```

The architecture uses:

- token embeddings
- positional embeddings
- causal masked self-attention
- multi-head attention
- feed-forward layers
- layer normalization
- dropout

## Checkpoint

`LLM.py` saves:

```text
byte_pair_attention_model_vocab_200.pth
```

Although the extension is `.pth`, this file is a checkpoint dictionary. It contains:

```text
model_state_dict
stoi
itos
vocab_size
merges
base_vocab
block_size
n_embd
n_head
n_layer
dropout
```

This makes the checkpoint usable for generation because it stores both the model weights and the tokenizer/model configuration.

## Train the Model

From the repository root:

```bash
python LLM/LLM.py
```

Or from inside this folder:

```bash
cd LLM
python LLM.py
```

Before running on a new machine, update the `pd.read_csv(...)` path in `LLM.py` if needed.

## Generate From the Saved Checkpoint

Use the standalone generation script:

```bash
cd LLM
python generate_from_checkpoint.py \
  --checkpoint byte_pair_attention_model_vocab_200.pth \
  --num-smiles 1000
```

Useful options:

```bash
python generate_from_checkpoint.py --help
```

Common examples:

```bash
python generate_from_checkpoint.py --checkpoint byte_pair_attention_model_vocab_200.pth --num-smiles 10000
python generate_from_checkpoint.py --checkpoint byte_pair_attention_model_vocab_200.pth --num-smiles 1000 --temperature 0.8
python generate_from_checkpoint.py --checkpoint byte_pair_attention_model_vocab_200.pth --num-smiles 1000 --cpu
```

By default, duplicates are removed after canonicalization with RDKit. Use `--allow-duplicates` if duplicate canonical SMILES should be kept.

## Output

Generated SMILES are saved to:

```text
generated_smiles_from_ckpt.txt
```

The script prints:

- checkpoint path
- number of valid SMILES generated
- number of sampling attempts
- valid SMILES ratio
- output path

## Notes

- `LLM.py` is notebook-style Python and runs top-to-bottom.
- The generation section inside `LLM.py` currently has `generation_cycle = 0`; use `generate_from_checkpoint.py` for practical generation.
- `LLM.py` currently reads `tg_raw.csv` from an absolute local path. Update that path if running on another machine.
