
"""Compare TinyGPT training with original and expanded training data."""

import copy
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer
from utils.dataset import LanguageModelDataset


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORIGINAL_PATH = Path("data/raw/tiny_corpus.txt")
EXPANDED_PATH = Path("data/raw/tiny_corpus_expanded.txt")
OUTPUT_DIR = Path("experiments/dataset_comparison")

SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
NUM_MERGES = 25
PATIENCE = 5

MODEL_CONFIG = {
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "context_length": 128,
    "dropout": 0.2,
}


def set_seed():
    """Reset random generators for a reproducible experiment."""
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def read_paragraphs(path):
    """Read non-empty paragraphs from a UTF-8 corpus."""
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def make_loader(token_ids, shuffle):
    """Create a next-token prediction data loader."""
    dataset = LanguageModelDataset(token_ids, SEQUENCE_LENGTH)
    generator = torch.Generator().manual_seed(SEED) if shuffle else None

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def run_experiment(
    name,
    train_paragraphs,
    validation_paragraphs,
    shared_tokenizer,
):
    """Train one model using the same tokenizer as the other run."""
    set_seed()

    train_text = "\n\n".join(train_paragraphs)
    validation_text = "\n\n".join(validation_paragraphs)

    # Use a copy so both runs have exactly the same vocabulary and merges.
    tokenizer = copy.deepcopy(shared_tokenizer)

    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)

    if len(train_ids) <= SEQUENCE_LENGTH:
        raise ValueError(f"{name}: not enough training tokens.")

    if len(validation_ids) <= SEQUENCE_LENGTH:
        raise ValueError(f"{name}: not enough validation tokens.")

    train_loader = make_loader(train_ids, shuffle=True)
    validation_loader = make_loader(validation_ids, shuffle=False)

    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        **MODEL_CONFIG,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        """Calculate mean validation loss using the same procedure."""
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                loss = criterion(
                    logits.reshape(-1, len(tokenizer.vocab)),
                    y.reshape(-1),
                )

                total_loss += loss.item()
                total_batches += 1

        if total_batches == 0:
            raise ValueError("Validation loader is empty.")

        return total_loss / total_batches

    run_dir = OUTPUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_model.pt"
    tokenizer_path = run_dir / "tokenizer.json"

    fields = [
        "epoch",
        "train_loss",
        "validation_loss",
        "best_checkpoint",
    ]

    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, len(tokenizer.vocab)),
                y.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_batches += 1

        if total_batches == 0:
            raise ValueError("Training loader is empty.")

        train_loss = total_train_loss / total_batches
        validation_loss = evaluate()
        improved = validation_loss < best_loss

        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": MODEL_CONFIG.copy(),
                    "vocab_size": len(tokenizer.vocab),
                    "sequence_length": SEQUENCE_LENGTH,
                    "num_merges": NUM_MERGES,
                    "training_epoch": epoch,
                    "validation_loss": validation_loss,
                    "experiment_name": name,
                },
                checkpoint_path,
            )

            # Save the exact tokenizer paired with this checkpoint.
            tokenizer.save(tokenizer_path)
        else:
            stale_epochs += 1

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation_loss, 6),
            "best_checkpoint": improved,
        }
        history.append(record)

        print(
            f"{name} | Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train={train_loss:.4f} | val={validation_loss:.4f}"
            + (" | best" if improved else "")
        )

        # Save complete history after every epoch.
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)

        if stale_epochs >= PATIENCE:
            print(f"{name}: early stopping at epoch {epoch}.")
            break

    return {
        "experiment": name,
        "train_paragraphs": len(train_paragraphs),
        "validation_paragraphs": len(validation_paragraphs),
        "training_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "vocabulary_size": len(tokenizer.vocab),
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_loss, 6),
        "checkpoint": str(checkpoint_path),
    }


def main():
    original = read_paragraphs(ORIGINAL_PATH)
    additional = read_paragraphs(EXPANDED_PATH)

    if len(original) < 4:
        raise ValueError(
            "The original corpus needs at least four paragraphs."
        )

    # Preserve the original final three paragraphs as validation.
    original_train = original[:-3]
    shared_validation = original[-3:]

    validation_keys = {
        " ".join(p.casefold().split())
        for p in shared_validation
    }

    # Do not add exact copies of validation paragraphs to training.
    additional = [
        p
        for p in additional
        if " ".join(p.casefold().split()) not in validation_keys
    ]

    expanded_train = original_train + additional

    # Train the tokenizer only on original training paragraphs.
    # Both experiments use exactly this vocabulary and these BPE merges.
    shared_tokenizer = BPETokenizer()
    shared_tokenizer.train(
        "\n\n".join(original_train),
        num_merges=NUM_MERGES,
    )

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Shared validation paragraphs:", len(shared_validation))
    print("Original training paragraphs:", len(original_train))
    print("Expanded training paragraphs:", len(expanded_train))
    print("Shared vocabulary size:", len(shared_tokenizer.vocab))
    print("Dropout:", MODEL_CONFIG["dropout"])
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        run_experiment(
            "original",
            original_train,
            shared_validation,
            shared_tokenizer,
        ),
        run_experiment(
            "expanded",
            expanded_train,
            shared_validation,
            shared_tokenizer,
        ),
    ]

    summary_path = OUTPUT_DIR / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Dataset Comparison Summary ===")

    for result in results:
        print(
            f"{result['experiment']}: "
            f"best val loss={result['best_validation_loss']:.4f}, "
            f"vocab={result['vocabulary_size']}, "
            f"train tokens={result['training_tokens']}"
        )

    print("Summary saved to:", summary_path)
    print("Main checkpoint and tokenizer were not modified.")


if __name__ == "__main__":
    main()
