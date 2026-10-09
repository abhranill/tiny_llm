
"""Dataset utilities for our tiny language model."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer.bpe import BPETokenizer


def load_corpus(path):
    """Read the training corpus from a UTF-8 text file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"The corpus file is empty: {path}")

    return text


class LanguageModelDataset(Dataset):
    """Return input and next-token target sequences."""

    def __init__(self, token_ids, sequence_length):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        self.sequence_length = sequence_length

        if len(self.token_ids) <= sequence_length:
            raise ValueError(
                "Not enough tokens for this sequence length."
            )

    def __len__(self):
        # Every starting position produces one training example.
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index):
        # Take sequence_length tokens for the input.
        x = self.token_ids[
            index : index + self.sequence_length
        ]

        # Shift the same sequence one position to the right.
        y = self.token_ids[
            index + 1 : index + self.sequence_length + 1
        ]

        return x, y


def prepare_datasets(
    text,
    sequence_length=16,
    train_ratio=0.9,
    num_merges=100,
):
    """Train the tokenizer and create train/validation datasets."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    # Train our custom tokenizer on the corpus.
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=num_merges)

    # Convert the complete corpus into integer token IDs.
    token_ids = tokenizer.encode(text)

    # Split the token stream into training and validation parts.
    split_index = int(len(token_ids) * train_ratio)

    train_ids = token_ids[:split_index]
    val_ids = token_ids[split_index:]

    # Construct separate datasets to avoid windows crossing
    # from training data into validation data.
    train_dataset = LanguageModelDataset(
        train_ids,
        sequence_length,
    )

    val_dataset = LanguageModelDataset(
        val_ids,
        sequence_length,
    )

    return tokenizer, train_dataset, val_dataset
