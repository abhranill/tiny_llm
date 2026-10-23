
"""Run an isolated, reproducible TinyGPT training experiment."""

import csv
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer
from utils.dataset import LanguageModelDataset, load_corpus


SEED = 42
CORPUS_PATH = Path("data/raw/tiny_corpus.txt")
OUTPUT_DIR = Path("experiments/reproducible_training_repeat")

SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
NUM_MERGES = 25
TRAIN_RATIO = 0.85
PATIENCE = 5

MODEL_CONFIG = {
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "context_length": 128,
    "dropout": 0.0,
}


def set_seed(seed):
    """Set random seeds and request deterministic CUDA behavior."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path):
    """Calculate a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def evaluate(model, loader, criterion, device, vocab_size):
    """Calculate token-weighted validation loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

            token_count = y.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return total_loss / total_tokens


def main():
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    corpus = load_corpus(str(CORPUS_PATH))

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

    # Train the tokenizer on training paragraphs only.
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, num_merges=NUM_MERGES)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    if len(train_ids) <= SEQUENCE_LENGTH:
        raise ValueError("Not enough training tokens.")

    if len(val_ids) <= SEQUENCE_LENGTH:
        raise ValueError("Not enough validation tokens.")

    train_dataset = LanguageModelDataset(
        train_ids, SEQUENCE_LENGTH
    )
    val_dataset = LanguageModelDataset(
        val_ids, SEQUENCE_LENGTH
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        **MODEL_CONFIG,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    history_path = OUTPUT_DIR / "history.csv"
    metadata_path = OUTPUT_DIR / "metadata.json"

    corpus_hash = sha256_file(CORPUS_PATH)

    config = {
        "seed": SEED,
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_hash,
        "train_ratio": TRAIN_RATIO,
        "training_paragraphs": split_index,
        "validation_paragraphs": len(paragraphs) - split_index,
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_epochs": NUM_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_merges": NUM_MERGES,
        "model_config": MODEL_CONFIG,
        "vocab_size": len(tokenizer.vocab),
        "device": str(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "deterministic_algorithms": True,
    }

    history_fields = [
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "best_checkpoint",
    ]

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("PyTorch version:", torch.__version__)
    print("Corpus SHA-256:", corpus_hash)
    print("Training paragraphs:", split_index)
    print("Validation paragraphs:", len(paragraphs) - split_index)
    print("Training tokens:", len(train_ids))
    print("Validation tokens:", len(val_ids))
    print("Vocabulary size:", len(tokenizer.vocab))
    print("Deterministic algorithms: enabled")
    print("Experiment output:", OUTPUT_DIR)

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=history_fields
        )
        writer.writeheader()

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_train_loss = 0.0
            total_train_tokens = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)

                loss = criterion(
                    logits.reshape(-1, len(tokenizer.vocab)),
                    y.reshape(-1),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

                token_count = y.numel()
                total_train_loss += loss.item() * token_count
                total_train_tokens += token_count

            if total_train_tokens == 0:
                raise ValueError("Training loader is empty.")

            train_loss = (
                total_train_loss / total_train_tokens
            )
            val_loss = evaluate(
                model,
                val_loader,
                criterion,
                device,
                len(tokenizer.vocab),
            )

            improved = val_loss < best_validation_loss

            if improved:
                best_validation_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # Keep experimental artifacts together and separate
                # from the project's baseline checkpoint and tokenizer.
                tokenizer.save(tokenizer_path)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": MODEL_CONFIG,
                        "vocab_size": len(tokenizer.vocab),
                        "sequence_length": SEQUENCE_LENGTH,
                        "num_merges": NUM_MERGES,
                        "training_epoch": epoch + 1,
                        "best_epoch": best_epoch,
                        "validation_loss": val_loss,
                        "seed": SEED,
                        "corpus_sha256": corpus_hash,
                        "training_config": config,
                        "optimizer_config": {
                            "name": "AdamW",
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                        },
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "validation_loss": round(val_loss, 6),
                "learning_rate": LEARNING_RATE,
                "best_checkpoint": improved,
            }

            history.append(record)
            writer.writerow(record)
            file.flush()

            print(
                f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
                f"train={train_loss:.4f} | val={val_loss:.4f}"
                + (" | best" if improved else "")
            )

            if epochs_without_improvement >= PATIENCE:
                print(
                    "Early stopping after "
                    f"{PATIENCE} non-improving epochs."
                )
                break

    config["epochs_completed"] = len(history)
    config["best_epoch"] = best_epoch
    config["best_validation_loss"] = best_validation_loss

    metadata_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment complete.")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print("Checkpoint:", checkpoint_path)
    print("Tokenizer:", tokenizer_path)
    print("History:", history_path)
    print("Metadata:", metadata_path)
    print("Main checkpoint and tokenizer were not modified.")


if __name__ == "__main__":
    main()