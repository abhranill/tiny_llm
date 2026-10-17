
"""Compare TinyGPT dropout settings without overwriting the main checkpoint."""

import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer.bpe import BPETokenizer
from utils.dataset import LanguageModelDataset, load_corpus
from model.transformer import TinyGPT


# Experiment configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
NUM_MERGES = 25
TRAIN_RATIO = 0.85
PATIENCE = 5
WEIGHT_DECAY = 0.01

BASE_MODEL_CONFIG = {
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "context_length": 128,
}

EXPERIMENTS = [
    {"name": "baseline", "dropout": 0.0},
    {"name": "dropout_01", "dropout": 0.1},
    {"name": "dropout_02", "dropout": 0.2},
]

OUTPUT_DIR = Path("experiments/regularization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "results.csv"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    """Create one fixed paragraph split and tokenizer for all experiments."""
    corpus = load_corpus("data/raw/tiny_corpus.txt")

    paragraphs = [
        paragraph.strip()
        for paragraph in corpus.split("\n\n")
        if paragraph.strip()
    ]

    if len(paragraphs) < 2:
        raise ValueError("Need at least two paragraphs for a split.")

    split_index = int(len(paragraphs) * TRAIN_RATIO)
    split_index = max(1, min(split_index, len(paragraphs) - 1))

    train_text = "\n\n".join(paragraphs[:split_index])
    val_text = "\n\n".join(paragraphs[split_index:])

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, num_merges=NUM_MERGES)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    if len(train_ids) <= SEQUENCE_LENGTH:
        raise ValueError("Not enough training tokens.")

    if len(val_ids) <= SEQUENCE_LENGTH:
        raise ValueError("Not enough validation tokens.")

    train_dataset = LanguageModelDataset(train_ids, SEQUENCE_LENGTH)
    val_dataset = LanguageModelDataset(val_ids, SEQUENCE_LENGTH)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Training paragraphs: {split_index}")
    print(f"Validation paragraphs: {len(paragraphs) - split_index}")
    print(f"Training tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(val_ids)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    return (
        len(tokenizer.vocab),
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
    )


def evaluate(model, loader, criterion, vocab_size):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return total_loss / total_batches


def run_experiment(experiment, vocab_size, train_loader, val_loader):
    # Reset model initialization and data-loader randomness for a fairer
    # comparison. Dropout itself will consume random numbers during training.
    set_seed(SEED)

    config = {
        **BASE_MODEL_CONFIG,
        "dropout": experiment["dropout"],
    }

    model = TinyGPT(vocab_size=vocab_size, **config).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_train_loss = float("inf")

    checkpoint_path = OUTPUT_DIR / f"{experiment['name']}.pt"
    history_path = OUTPUT_DIR / f"{experiment['name']}_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "dropout",
                "weight_decay",
            ],
        )
        writer.writeheader()

    print(f"\n--- Experiment: {experiment['name']} ---")
    print(f"Dropout: {experiment['dropout']}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_batches += 1

        if total_batches == 0:
            raise ValueError("Cannot train with an empty data loader.")

        train_loss = total_train_loss / total_batches
        val_loss = evaluate(model, val_loader, criterion, vocab_size)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_train_loss = train_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "vocab_size": vocab_size,
                    "sequence_length": SEQUENCE_LENGTH,
                    "num_merges": NUM_MERGES,
                    "training_epoch": best_epoch,
                    "validation_loss": best_val_loss,
                    "weight_decay": WEIGHT_DECAY,
                    "experiment_name": experiment["name"],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        with history_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "validation_loss",
                    "dropout",
                    "weight_decay",
                ],
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "validation_loss": round(val_loss, 6),
                    "dropout": experiment["dropout"],
                    "weight_decay": WEIGHT_DECAY,
                }
            )

        print(
            f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            + (" | Best saved" if improved else "")
        )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    result = {
        "experiment": experiment["name"],
        "dropout": experiment["dropout"],
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_train_loss": round(best_train_loss, 6),
        "best_validation_loss": round(best_val_loss, 6),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    print(
        f"Best epoch: {best_epoch} | "
        f"Best validation loss: {best_val_loss:.4f}"
    )

    return result


def main():
    vocab_size, train_loader, val_loader = load_data()
    results = []

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment,
            vocab_size,
            train_loader,
            val_loader,
        )
        results.append(result)

    fields = [
        "experiment",
        "dropout",
        "weight_decay",
        "best_epoch",
        "best_train_loss",
        "best_validation_loss",
        "checkpoint",
        "history",
    ]

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Regularization comparison ===")
    for result in sorted(
        results,
        key=lambda item: item["best_validation_loss"],
    ):
        print(
            f"{result['experiment']:12s} | "
            f"dropout={result['dropout']:.1f} | "
            f"best epoch={result['best_epoch']:2d} | "
            f"validation loss={result['best_validation_loss']:.4f}"
        )

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("Main checkpoint and tokenizer were not overwritten.")


if __name__ == "__main__":
    main()
