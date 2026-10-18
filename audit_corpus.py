"""Audit a TinyGPT corpus for basic quality issues."""

import sys
from pathlib import Path

DEFAULT_PATH = Path("data/raw/tiny_corpus.txt")


def main():
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    text = corpus_path.read_text(encoding="utf-8")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    normalized = [paragraph.casefold() for paragraph in paragraphs]
    duplicates = sorted({
        paragraph
        for paragraph in normalized
        if normalized.count(paragraph) > 1
    })

    word_counts = [len(paragraph.split()) for paragraph in paragraphs]

    print("=== TinyGPT Corpus Audit ===")
    print("File:", corpus_path)
    print("Total characters:", len(text))
    print("Total words:", len(text.split()))
    print("Paragraph count:", len(paragraphs))
    print("Average paragraph words:", round(sum(word_counts) / len(word_counts), 2)
          if word_counts else 0)
    print("Shortest paragraph words:", min(word_counts, default=0))
    print("Duplicate paragraph groups:", len(duplicates))

    short_paragraphs = [
        (index, count, paragraphs[index - 1][:80])
        for index, count in enumerate(word_counts, start=1)
        if count < 30
    ]

    print("\nParagraphs under 30 words:", len(short_paragraphs))
    for index, count, preview in short_paragraphs:
        print(f"  Paragraph {index}: {count} words — {preview}")

    if duplicates:
        print("\nDuplicate paragraph previews:")
        for paragraph in duplicates:
            print(" ", paragraph[:100])

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()