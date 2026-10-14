
"""Evaluate TinyGPT using non-overlapping next-token targets."""

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
        raise ValueError("Tokenizer does not match checkpoint vocabulary.")

    corpus = load_corpus(CORPUS_PATH)
    paragraphs = [
        p.strip() for p in corpus.split("\n\n") if p.strip()
    ]

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

    context_length = checkpoint["model_config"]["context_length"]
    total_nll = 0.0
    total_targets = 0

    # Score each target token once, using non-overlapping chunks.
    # Each chunk predicts the tokens immediately following its inputs.
    with torch.no_grad():
        for start in range(0, len(validation_ids) - 1, context_length):
            end = min(start + context_length, len(validation_ids) - 1)

            input_chunk = validation_ids[start:end]
            target_chunk = validation_ids[start + 1:end + 1]

            if not input_chunk:
                continue

            x = torch.tensor(
                [input_chunk], dtype=torch.long, device=DEVICE
            )
            y = torch.tensor(
                [target_chunk], dtype=torch.long, device=DEVICE
            )

            logits = model(x)

            nll = F.cross_entropy(
                logits.reshape(-1, checkpoint["vocab_size"]),
                y.reshape(-1),
                reduction="sum",
            )

            total_nll += nll.item()
            total_targets += y.numel()

    mean_loss = total_nll / total_targets
    perplexity = math.exp(mean_loss) if mean_loss < 709 else float("inf")

    print("\n--- TinyGPT Evaluation Report ---")
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint epoch:", checkpoint.get("training_epoch", "unknown"))
    print("Checkpoint validation loss:", checkpoint.get("validation_loss", "unknown"))
    print("Vocabulary size:", checkpoint["vocab_size"])
    print("Validation paragraphs:", len(paragraphs) - split_index)
    print("Validation tokens:", len(validation_ids))
    print("Unique next-token targets evaluated:", total_targets)
    print(f"Mean validation loss: {mean_loss:.4f}")
    print(f"Validation perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
