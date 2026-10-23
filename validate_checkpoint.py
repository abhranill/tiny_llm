"""Validate TinyGPT checkpoint integrity without modifying artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer


DEFAULT_CHECKPOINT = Path("checkpoints/tiny_gpt.pt")
DEFAULT_TOKENIZER = Path("tokenizer/tokenizer.json")
DEFAULT_CORPUS = Path("data/raw/tiny_corpus.txt")

REQUIRED_KEYS = {
    "model_config",
    "model_state_dict",
    "vocab_size",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    """Safely load checkpoint data with a narrowly allowlisted metadata type."""
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )



def validate(checkpoint_path, tokenizer_path, corpus_path):
    results = []

    def record(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results.append((status, name, detail))

    paths = {
        "checkpoint": Path(checkpoint_path),
        "tokenizer": Path(tokenizer_path),
        "corpus": Path(corpus_path),
    }

    for name, path in paths.items():
        record(
            f"{name} file exists",
            path.is_file(),
            str(path),
        )

    if not all(path.is_file() for path in paths.values()):
        return results

    try:
        checkpoint = load_checkpoint(paths["checkpoint"])
        record("Safe checkpoint loading", True, "weights_only=True")
    except Exception as exc:
        record(
            "Safe checkpoint loading",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return results

    record(
        "Checkpoint is a dictionary",
        isinstance(checkpoint, dict),
        type(checkpoint).__name__,
    )
    if not isinstance(checkpoint, dict):
        return results

    missing = sorted(REQUIRED_KEYS - checkpoint.keys())
    record(
        "Required checkpoint keys",
        not missing,
        f"Missing: {missing}" if missing else "All present",
    )
    if missing:
        return results

    vocab_size = checkpoint["vocab_size"]
    model_config = checkpoint["model_config"]
    state_dict = checkpoint["model_state_dict"]

    record(
        "Vocabulary size is a positive integer",
        isinstance(vocab_size, int)
        and not isinstance(vocab_size, bool)
        and vocab_size > 0,
        repr(vocab_size),
    )
    record(
        "Model configuration is a dictionary",
        isinstance(model_config, dict),
        type(model_config).__name__,
    )
    record(
        "Model state is a dictionary",
        isinstance(state_dict, dict),
        type(state_dict).__name__,
    )

    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
        or not isinstance(model_config, dict)
        or not isinstance(state_dict, dict)
    ):
        return results

    try:
        tokenizer = BPETokenizer()
        tokenizer.load(str(paths["tokenizer"]))
        tokenizer_vocab_size = len(tokenizer.vocab)
        record(
            "Checkpoint/tokenizer vocabulary match",
            vocab_size == tokenizer_vocab_size,
            f"checkpoint={vocab_size}, tokenizer={tokenizer_vocab_size}",
        )
    except Exception as exc:
        record(
            "Tokenizer loads successfully",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return results

    try:
        model = TinyGPT(vocab_size=vocab_size, **model_config)
        model.load_state_dict(state_dict, strict=True)
        record(
            "Model reconstruction and state compatibility",
            True,
            "Strict state-dict loading succeeded",
        )
    except Exception as exc:
        record(
            "Model reconstruction and state compatibility",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    current_corpus_hash = sha256_file(paths["corpus"])
    saved_hash = checkpoint.get("corpus_sha256")
    training_config = checkpoint.get("training_config")

    
    if saved_hash is None:
        print(
            "WARNING: Checkpoint has no corpus_sha256; "
            "historical corpus integrity cannot be verified."
        )
    else:
        record(
            "Checkpoint corpus SHA-256",
            saved_hash == current_corpus_hash,
            f"saved={saved_hash}, current={current_corpus_hash}",
        )

    

    nested_hash = (
        training_config.get("corpus_sha256")
        if isinstance(training_config, dict)
        else None
    )
    if nested_hash is not None:
        record(
            "Training metadata corpus SHA-256",
            nested_hash == current_corpus_hash,
            f"saved={nested_hash}, current={current_corpus_hash}",
        )

    for key in ("seed", "training_config", "optimizer_config", "best_epoch"):
        if key not in checkpoint:
            print(f"WARNING: Missing optional metadata: {key}")

    print("\nTinyGPT checkpoint integrity report")
    print("=" * 72)
    for status, name, detail in results:
        print(f"[{status}] {name}")
        if detail:
            print(f"       {detail}")

    print("\nTokenizer identity limitation:")
    print("Matching vocabulary sizes alone do not prove tokenizer identity.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate TinyGPT checkpoint, tokenizer, and corpus."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    results = validate(args.checkpoint, args.tokenizer, args.corpus)

    print("\nSummary:")
    passed = sum(status == "PASS" for status, _, _ in results)
    failed = sum(status == "FAIL" for status, _, _ in results)
    print(f"PASS: {passed} | FAIL: {failed}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
