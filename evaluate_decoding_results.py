
"""Calculate simple diversity and repetition metrics for decoding results."""

import csv
from collections import Counter
from pathlib import Path


INPUT_PATH = Path("experiments/decoding_comparison/results.csv")
OUTPUT_PATH = Path("experiments/decoding_comparison/metrics.csv")


def repeated_ngram_rate(tokens, n):
    """Fraction of n-gram occurrences that repeat an earlier n-gram."""
    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The results CSV contains no generation rows.")

    metrics = []

    for row in rows:
        prompt = row["prompt"]
        generated_text = row["generated_text"]

        # Measure the continuation rather than counting the prompt itself.
        if generated_text.startswith(prompt):
            continuation = generated_text[len(prompt):].strip()
        else:
            continuation = generated_text.strip()

        tokens = continuation.split()
        token_count = len(tokens)
        unique_token_ratio = (
            len(set(tokens)) / token_count if token_count else 0.0
        )

        metrics.append({
            "prompt": prompt,
            "strategy": row["strategy"],
            "whitespace_token_count": token_count,
            "unique_token_ratio": round(unique_token_ratio, 4),
            "repeated_bigram_rate": round(repeated_ngram_rate(tokens, 2), 4),
            "repeated_trigram_rate": round(repeated_ngram_rate(tokens, 3), 4),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print("Generation quality metrics")
    print("=" * 75)

    for metric in metrics:
        print(
            f"{metric['prompt']!r:22} | "
            f"{metric['strategy']:16} | "
            f"tokens={metric['whitespace_token_count']:2} | "
            f"unique={metric['unique_token_ratio']:.3f} | "
            f"repeat-2gram={metric['repeated_bigram_rate']:.3f} | "
            f"repeat-3gram={metric['repeated_trigram_rate']:.3f}"
        )

    print("\nMetrics saved to:", OUTPUT_PATH)
    print(
        "Note: these are simple whitespace-based diagnostics, "
        "not measures of semantic quality."
    )


if __name__ == "__main__":
    main()
