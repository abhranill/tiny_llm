
"""Train TinyGPT on our small text corpus."""

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

MODEL_CONFIG = {
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "context_length": 128,
    "dropout": 0.0,
}


# 2. Load and split the text before training the tokenizer
corpus = load_corpus("data/raw/tiny_corpus.txt")

# Split the corpus by paragraph to avoid cutting text mid-paragraph.
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


# 3. Create the model and optimizer
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


# 4. Evaluate without updating the model
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

    model.train()
    return total_loss / total_batches



# 5. Training loop with early stopping
best_validation_loss = float("inf")
patience = 5
epochs_without_improvement = 0

checkpoint_path = Path("checkpoints/tiny_gpt.pt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0

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

    average_train_loss = total_train_loss / len(train_loader)
    validation_loss = evaluate(val_loader)

    improved = validation_loss < best_validation_loss

    if improved:
        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        # Save the best weights seen so far.
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

    print(
        f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
        f"Train loss: {average_train_loss:.4f} | "
        f"Validation loss: {validation_loss:.4f}"
        + (" | Best checkpoint saved" if improved else "")
    )

    if epochs_without_improvement >= patience:
        print("Early stopping: validation loss did not improve.")
        break

print("Training complete!")
print(f"Best validation loss: {best_validation_loss:.4f}")
print("Best checkpoint saved to:", checkpoint_path)
print("Tokenizer saved to: tokenizer/tokenizer.json")


