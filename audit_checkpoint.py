
"""Audit TinyGPT checkpoint metadata without modifying project files."""

import hashlib
from pathlib import Path

import torch

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT_PATH = Path("checkpoints/tiny_gpt.pt")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
CORPUS_PATH = Path("data/raw/tiny_corpus.txt")

RECOMMENDED_METADATA = [
    "seed",
    "corpus_sha256",
    "training_config",
    "optimizer_config",
    "best_epoch",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    for path in (CHECKPOINT_PATH, TOKENIZER_PATH, CORPUS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    print("=" * 60)
    print("TINYGPT CHECKPOINT AUDIT")
    print("=" * 60)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Tokenizer:", TOKENIZER_PATH)
    print("Corpus:", CORPUS_PATH)

    print("\nCheckpoint metadata keys:")
    for key in sorted(checkpoint):
        if key != "model_state_dict":
            print(f"  {key}: {checkpoint[key]}")

    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    saved_vocab_size = checkpoint.get("vocab_size")
    tokenizer_vocab_size = len(tokenizer.vocab)

    print("\nCompatibility checks:")
    print("  Checkpoint vocabulary size:", saved_vocab_size)
    print("  Tokenizer vocabulary size:", tokenizer_vocab_size)
    print("  Vocabulary-size match:", saved_vocab_size == tokenizer_vocab_size)

    model = TinyGPT(
        vocab_size=saved_vocab_size,
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print("  Model reconstruction: PASSED")

    corpus_hash = sha256_file(CORPUS_PATH)
    print("\nCurrent corpus SHA-256:")
    print(" ", corpus_hash)

    missing = [
        key for key in RECOMMENDED_METADATA
        if key not in checkpoint
    ]

    print("\nRecommended metadata missing from checkpoint:")
    if missing:
        for key in missing:
            print("  -", key)
    else:
        print("  None")

    print("\nImportant limitation:")
    print(
        "Matching vocabulary sizes do not prove that this tokenizer "
        "is the exact tokenizer used to train the checkpoint."
    )
    print(
        "The corpus hash above identifies the current corpus only; "
        "it does not establish which corpus produced this checkpoint."
    )

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()
