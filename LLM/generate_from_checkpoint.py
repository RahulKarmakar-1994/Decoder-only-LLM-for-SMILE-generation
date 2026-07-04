#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature):
        idx_prev = idx[:, -1:]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_prev.item() != 0 and idx_next.item() == 0:
                break

            idx_prev = idx_next
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def decode_tokens(token_ids, itos):
    return "".join(itos[int(token_id)] for token_id in token_ids)


def generate_valid_smiles(
    model,
    itos,
    device,
    num_smiles,
    max_attempts,
    max_new_tokens,
    temperature,
    keep_unique,
):
    valid_smiles = []
    seen_smiles = set()

    with torch.no_grad():
        for attempt in range(1, max_attempts + 1):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            token_ids = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )[0].tolist()

            token_ids = [token_id for token_id in token_ids if token_id != 0]
            if not token_ids:
                continue

            smiles = decode_tokens(token_ids, itos)
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue

            canonical_smiles = Chem.MolToSmiles(molecule)
            if keep_unique and canonical_smiles in seen_smiles:
                continue

            seen_smiles.add(canonical_smiles)
            valid_smiles.append(canonical_smiles)

            if len(valid_smiles) % 50 == 0:
                print(f"Generated {len(valid_smiles)} valid SMILES after {attempt} attempts")

            if len(valid_smiles) >= num_smiles:
                return valid_smiles, attempt

    return valid_smiles, max_attempts


def resolve_checkpoint_path(checkpoint_path):
    if checkpoint_path.exists():
        return checkpoint_path

    cwd_checkpoint_path = Path.cwd() / checkpoint_path.name
    if cwd_checkpoint_path.exists():
        return cwd_checkpoint_path

    return checkpoint_path


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate valid SMILES from a saved BPE transformer checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=script_dir / "bpe_transformer_vocab200.ckpt",
        help="Path to the .ckpt file saved by LLM.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "generated_smiles_from_ckpt.txt",
        help="Text file where generated SMILES will be saved.",
    )
    parser.add_argument(
        "--num-smiles",
        type=int,
        default=1000,
        help="Number of valid SMILES to generate.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum sampling attempts. Default is num-smiles * 50.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=314,
        help="Maximum tokens to generate for one SMILES.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate canonical SMILES in the output.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)

    model = BigramLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = args.num_smiles * 50

    valid_smiles, attempts = generate_valid_smiles(
        model=model,
        itos=checkpoint["itos"],
        device=device,
        num_smiles=args.num_smiles,
        max_attempts=max_attempts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        keep_unique=not args.allow_duplicates,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as file:
        for smiles in valid_smiles:
            file.write(f"{smiles}\n")

    valid_percent = len(valid_smiles) / attempts if attempts > 0 else 0
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated {len(valid_smiles)} valid SMILES after {attempts} attempts")
    print(f"Valid SMILES %: {valid_percent}")
    print(f"Saved generated SMILES at: {args.output}")


if __name__ == "__main__":
    main()
