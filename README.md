# Decoder-only LLM for SMILES Generation

This repository contains three autoregressive SMILES-generation experiments built around the same polymer SMILES dataset:

- `LLM/`: decoder-only Transformer with causal self-attention and BPE-style token merging.
- `N-gram/`: feed-forward n-gram neural language model.
- `Wavenet/`: WaveNet-style hierarchical MLP with consecutive-token flattening and batch normalization.

Each method tokenizes SMILES strings, trains a next-token model, samples new SMILES, validates generated strings with RDKit, and compares generated molecules against the training set using canonical SMILES.

## Repository Layout

```text
.
├── LLM/
│   ├── LLM.py
│   ├── generate_from_checkpoint.py
│   ├── byte_pair_attention_model_vocab_200.pth
│   ├── tg_raw.csv
│   └── README.md
├── N-gram/
│   ├── n-gram_w_batchnorm_block_size_varry.py
│   └── README.md
├── Wavenet/
│   ├── Wavenet.py
│   └── README.md
└── README.md
```

## Requirements

The scripts use:

- Python 3
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- RDKit

Example Conda setup:

```bash
conda create -n smiles-llm python=3.10
conda activate smiles-llm
conda install -c conda-forge rdkit pandas numpy scikit-learn matplotlib seaborn
pip install torch
```

## Dataset

The scripts expect a CSV file containing a `SMILES` column. The included example dataset is:

```text
LLM/tg_raw.csv
```

Some training scripts currently contain absolute local paths to `tg_raw.csv`. If running on a different machine, update the path inside the script or run from an environment where that path exists.

## Quick Generation From Checkpoint

The `LLM/` folder includes a checkpoint file:

```text
LLM/byte_pair_attention_model_vocab_200.pth
```

Generate valid SMILES from it:

```bash
cd LLM
python generate_from_checkpoint.py \
  --checkpoint byte_pair_attention_model_vocab_200.pth \
  --num-smiles 1000
```

The generated SMILES are written to:

```text
LLM/generated_smiles_from_ckpt.txt
```

## Run the Training Scripts

From the repository root:

```bash
python LLM/LLM.py
python N-gram/n-gram_w_batchnorm_block_size_varry.py
python Wavenet/Wavenet.py
```

The scripts are notebook-style Python files and run top-to-bottom. Before running on a new machine, check the `pd.read_csv(...)` path inside each script and point it to your local `tg_raw.csv`.

## Method READMEs

See the method-specific documentation for details:

- [LLM/README.md](LLM/README.md)
- [N-gram/README.md](N-gram/README.md)
- [Wavenet/README.md](Wavenet/README.md)
