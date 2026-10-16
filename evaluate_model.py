
"""Evaluate TinyGPT with consistent, non-overlapping target chunks."""

import math

import torch
import torch.nn.functional as F

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer
from utils.dataset import load_corpus


CHECKPOINT_PATH = "checkpoints/tiny_gpt.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
CORPUS_PATH = "data/raw/tiny_corpus.txt"
TRAIN_RATIO = 0.85

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    if len(tokenizer.vocab) != checkpoint["vocab_size"]:
        raise ValueError(
            "Tokenizer vocabulary does not match checkpoint."
        )

    corpus = load_corpus(CORPUS_PATH)
    paragraphs = [
        p.strip()
        for p in corpus.split("\n\n")
        if p.strip()
    ]

    if len(paragraphs) < 2:
        raise ValueError("At least two paragraphs are required.")

    split_index = int(len(paragraphs) * TRAIN_RATIO)
    split_index = max(1, min(split_index, len(paragraphs) - 1))

    validation_text = "\n\n".join(paragraphs[split_index:])
    validation_ids = tokenizer.encode(validation_text)

    if len(validation_ids) < 2:
        raise ValueError("Not enough validation tokens.")

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use the sequence length used to train this checkpoint.
    chunk_length = checkpoint["sequence_length"]

    total_nll = 0.0
    total_targets = 0

    with torch.no_grad():
        # Every next-token target is evaluated exactly once.
        for start in range(0, len(validation_ids) - 1, chunk_length):
            end = min(start + chunk_length, len(validation_ids) - 1)

            input_chunk = validation_ids[start:end]
            target_chunk = validation_ids[start + 1:end + 1]

            x = torch.tensor(
                [input_chunk],
                dtype=torch.long,
                device=DEVICE,
            )
            y = torch.tensor(
                [target_chunk],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(x)

            loss_sum = F.cross_entropy(
                logits.reshape(-1, checkpoint["vocab_size"]),
                y.reshape(-1),
                reduction="sum",
            )

            total_nll += loss_sum.item()
            total_targets += y.numel()

    mean_loss = total_nll / total_targets
    perplexity = (
        math.exp(mean_loss)
        if mean_loss < 709
        else float("inf")
    )

    print("\n--- TinyGPT Evaluation Report ---")
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(
        "Checkpoint epoch:",
        checkpoint.get("training_epoch", "unknown"),
    )
    print("Training sequence length:", chunk_length)
    print("Vocabulary size:", checkpoint["vocab_size"])
    print("Validation paragraphs:", len(paragraphs) - split_index)
    print("Validation tokens:", len(validation_ids))
    print("Unique next-token targets evaluated:", total_targets)
    print(f"Mean validation loss: {mean_loss:.4f}")
    print(f"Validation perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
