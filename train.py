


import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer.bpe import BPETokenizer
from utils.dataset import LanguageModelDataset, load_corpus
from model.transformer import TinyGPT


# 1. Configuration
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
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


# 2. Load and split the corpus by paragraph
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

print(f"Training paragraphs: {split_index}")
print(f"Validation paragraphs: {len(paragraphs) - split_index}")
print(f"Training characters: {len(train_text)}")
print(f"Validation characters: {len(val_text)}")


# 3. Train the tokenizer on training text only
tokenizer = BPETokenizer()
tokenizer.train(train_text, num_merges=NUM_MERGES)

train_ids = tokenizer.encode(train_text)
val_ids = tokenizer.encode(val_text)

if len(train_ids) <= SEQUENCE_LENGTH:
    raise ValueError("Not enough training tokens.")

if len(val_ids) <= SEQUENCE_LENGTH:
    raise ValueError(
        "Not enough validation tokens. Use a shorter sequence length "
        "or a larger corpus."
    )

train_dataset = LanguageModelDataset(train_ids, SEQUENCE_LENGTH)
val_dataset = LanguageModelDataset(val_ids, SEQUENCE_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Save the tokenizer used by this model.
tokenizer.save("tokenizer/tokenizer.json")


# 4. Create the model and optimizer
model = TinyGPT(
    vocab_size=len(tokenizer.vocab),
    **MODEL_CONFIG,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss()

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Corpus characters:", len(corpus))
print("Training tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))
print("Vocabulary size:", len(tokenizer.vocab))
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)


# 5. Evaluate without updating model parameters
def evaluate(data_loader):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in data_loader:
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
        raise ValueError("Cannot evaluate an empty data loader.")

    return total_loss / total_batches


# 6. Prepare checkpoint and training-history files
checkpoint_path = Path("checkpoints/tiny_gpt.pt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

history_path = Path("experiments/training_history.csv")
history_path.parent.mkdir(parents=True, exist_ok=True)

history_fields = [
    "epoch",
    "train_loss",
    "validation_loss",
    "learning_rate",
    "best_checkpoint",
]

# Start a fresh CSV for this training run.
with history_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=history_fields)
    writer.writeheader()

training_history = []


# 7. Training loop with early stopping and diagnostics
best_validation_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    total_train_batches = 0

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
        total_train_batches += 1

    if total_train_batches == 0:
        raise ValueError("Cannot train with an empty data loader.")

    average_train_loss = total_train_loss / total_train_batches
    validation_loss = evaluate(val_loader)

    improved = validation_loss < best_validation_loss
    current_lr = optimizer.param_groups[0]["lr"]

    if improved:
        best_validation_loss = validation_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        # Save the best model, not necessarily the final epoch.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": MODEL_CONFIG,
                "vocab_size": len(tokenizer.vocab),
                "sequence_length": SEQUENCE_LENGTH,
                "num_merges": NUM_MERGES,
                "training_epoch": epoch + 1,
                "validation_loss": validation_loss,
            },
            checkpoint_path,
        )
    else:
        epochs_without_improvement += 1

    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": round(average_train_loss, 6),
        "validation_loss": round(validation_loss, 6),
        "learning_rate": current_lr,
        "best_checkpoint": improved,
    }

    training_history.append(epoch_record)

    # Save each epoch immediately so history survives early stopping.
    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history_fields)
        writer.writerow(epoch_record)

    print(
        f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
        f"Train loss: {average_train_loss:.4f} | "
        f"Validation loss: {validation_loss:.4f} | "
        f"LR: {current_lr:.2e}"
        + (" | Best checkpoint saved" if improved else "")
    )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"Early stopping: validation loss did not improve "
            f"for {PATIENCE} consecutive epochs."
        )
        break


# 8. Final training report
print("\n" + "=" * 45)
print("TRAINING DIAGNOSTICS")
print("=" * 45)

best_record = min(
    training_history,
    key=lambda record: record["validation_loss"],
)

final_record = training_history[-1]

overfit_gap = (
    final_record["validation_loss"] - final_record["train_loss"]
)

print(f"Epochs completed: {len(training_history)}")
print(f"Best epoch: {best_record['epoch']}")
print(f"Best validation loss: {best_record['validation_loss']:.4f}")
print(f"Final training loss: {final_record['train_loss']:.4f}")
print(f"Final validation loss: {final_record['validation_loss']:.4f}")
print(f"Final validation-training gap: {overfit_gap:.4f}")

if len(training_history) >= 2:
    first_record = training_history[0]
    train_improvement = (
        first_record["train_loss"] - final_record["train_loss"]
    )
    val_change = (
        final_record["validation_loss"]
        - best_record["validation_loss"]
    )

    print(f"Training-loss reduction: {train_improvement:.4f}")
    print(f"Validation loss above best: {val_change:.4f}")

    if (
        final_record["train_loss"] < best_record["train_loss"]
        and final_record["validation_loss"] > best_record["validation_loss"]
    ):
        print(
            "Diagnostic: training loss improved while validation loss "
            "moved above its best value. Possible overfitting."
        )
    else:
        print(
            "Diagnostic: inspect the CSV history to assess the loss trends."
        )

print(f"Training history: {history_path}")
print(f"Best checkpoint: {checkpoint_path}")
print("Tokenizer: tokenizer/tokenizer.json")
print("=" * 45)
