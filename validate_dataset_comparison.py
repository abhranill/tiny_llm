
"""Validate the dataset-comparison split and tokenizer coverage."""

from pathlib import Path

from tokenizer.bpe import BPETokenizer


ORIGINAL_PATH = Path("data/raw/tiny_corpus.txt")
EXPANDED_PATH = Path("data/raw/tiny_corpus_expanded.txt")
TOKENIZER_PATH = Path(
    "experiments/dataset_comparison/original/tokenizer.json"
)


def read_paragraphs(path):
    return [
        paragraph.strip()
        for paragraph in path.read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip()
    ]


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    original = read_paragraphs(ORIGINAL_PATH)
    additional = read_paragraphs(EXPANDED_PATH)

    original_train = original[:-3]
    validation = original[-3:]

    validation_keys = {normalize(p) for p in validation}
    additional_train = [
        p for p in additional if normalize(p) not in validation_keys
    ]
    expanded_train = original_train + additional_train

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    unk_id = tokenizer.vocab[tokenizer.unk_token]

    additional_text = "\n\n".join(additional_train)
    additional_ids = tokenizer.encode(additional_text)

    unk_count = sum(token_id == unk_id for token_id in additional_ids)
    unk_percent = 100 * unk_count / len(additional_ids) if additional_ids else 0

    train_keys = {normalize(p) for p in expanded_train}
    overlap = train_keys.intersection(validation_keys)

    print("=== Dataset Comparison Validation ===")
    print("Original training paragraphs:", len(original_train))
    print("Additional training paragraphs:", len(additional_train))
    print("Shared validation paragraphs:", len(validation))
    print("Training/validation paragraph overlap:", len(overlap))
    print("Tokenizer vocabulary size:", len(tokenizer.vocab))
    print("Additional training tokens:", len(additional_ids))
    print("Unknown tokens in additional text:", unk_count)
    print(f"Unknown-token rate: {unk_percent:.2f}%")
    print("Validation overlap check:", "PASSED" if not overlap else "FAILED")

    if overlap:
        print("Overlapping paragraphs:")
        for paragraph in sorted(overlap):
            print(" ", paragraph[:100])

    print("\nValidation complete. No files were modified.")


if __name__ == "__main__":
    main()
