
"""Evaluate TinyGPT generations with token-aware and character metrics."""

import csv
from collections import Counter
from pathlib import Path

from tokenizer.bpe import BPETokenizer


INPUT_PATH = Path("experiments/decoding_comparison/results.csv")
OUTPUT_PATH = Path("experiments/decoding_comparison/quality_metrics.csv")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")


def repeated_ngram_rate(tokens, n):
    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def character_repetition_rate(text, n=4):
    """Fraction of character n-gram occurrences repeating earlier n-grams."""
    normalized = "".join(text.lower().split())
    ngrams = [
        normalized[i:i + n]
        for i in range(len(normalized) - n + 1)
    ]

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing generation results: {INPUT_PATH}")

    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    with INPUT_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No generation results found.")

    metrics = []

    for row in rows:
        prompt = row["prompt"]
        full_text = row["generated_text"]

        continuation = (
            full_text[len(prompt):].strip()
            if full_text.startswith(prompt)
            else full_text.strip()
        )

        token_ids = tokenizer.encode(continuation)
        unique_token_ratio = (
            len(set(token_ids)) / len(token_ids)
            if token_ids else 0.0
        )

        metrics.append({
            "prompt": prompt,
            "strategy": row["strategy"],
            "bpe_token_count": len(token_ids),
            "unique_bpe_token_ratio": round(unique_token_ratio, 4),
            "repeated_bpe_bigram_rate": round(
                repeated_ngram_rate(token_ids, 2), 4
            ),
            "repeated_bpe_trigram_rate": round(
                repeated_ngram_rate(token_ids, 3), 4
            ),
            "repeated_character_4gram_rate": round(
                character_repetition_rate(continuation, 4), 4
            ),
        })

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print("Tokenizer-aware generation quality metrics")
    print("=" * 100)

    for item in metrics:
        print(
            f"{item['prompt']!r:20} | {item['strategy']:16} | "
            f"BPE tokens={item['bpe_token_count']:2} | "
            f"unique={item['unique_bpe_token_ratio']:.3f} | "
            f"repeat-2gram={item['repeated_bpe_bigram_rate']:.3f} | "
            f"repeat-3gram={item['repeated_bpe_trigram_rate']:.3f} | "
            f"char-repeat={item['repeated_character_4gram_rate']:.3f}"
        )

    print("\nSaved to:", OUTPUT_PATH)
    print("These are diagnostics, not semantic-quality scores.")


if __name__ == "__main__":
    main()
