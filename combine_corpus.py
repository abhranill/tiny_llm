
"""Combine TinyGPT corpora while preserving the original files."""

from pathlib import Path

SOURCE_FILES = [
    Path("data/raw/tiny_corpus.txt"),
    Path("data/raw/tiny_corpus_expanded.txt"),
]
OUTPUT_FILE = Path("data/raw/tiny_corpus_combined.txt")


def main():
    paragraphs = []
    seen = set()

    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            key = " ".join(paragraph.casefold().split())
            if key not in seen:
                paragraphs.append(paragraph)
                seen.add(key)

    if len(paragraphs) < 2:
        raise ValueError("Combined corpus must contain at least two paragraphs.")

    combined_text = "\n\n".join(paragraphs) + "\n"
    OUTPUT_FILE.write_text(combined_text, encoding="utf-8")

    print("Combined corpus created:", OUTPUT_FILE)
    print("Source files:", len(SOURCE_FILES))
    print("Unique paragraphs:", len(paragraphs))
    print("Total words:", len(combined_text.split()))
    print("Total characters:", len(combined_text))
    print("Original source files were not modified.")


if __name__ == "__main__":
    main()
