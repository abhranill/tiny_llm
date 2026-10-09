
"""A small educational Byte Pair Encoding tokenizer."""

from collections import Counter
import json
from pathlib import Path


class BPETokenizer:
    """Learn frequent adjacent-token merges from text."""

    def __init__(self):
        # Each merge is stored as a pair: (left_token, right_token).
        self.merges = []

        # The vocabulary maps token strings to integer IDs.
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, text, num_merges=100):
        """Learn BPE merges from a training text."""

        if not text:
            raise ValueError("Training text cannot be empty.")

        # Initially, every Unicode character is one token.
        tokens = list(text)

        # Each iteration learns one new merge.
        for _ in range(num_merges):
            pair_counts = Counter(
                zip(tokens, tokens[1:])
            )

            # Stop if no adjacent pairs remain.
            if not pair_counts:
                break

            # Select the most frequent adjacent pair.
            best_pair, frequency = pair_counts.most_common(1)[0]

            # There is nothing useful to merge if all pairs
            # occur only once.
            if frequency < 2:
                break

            left, right = best_pair
            merged_token = left + right

            # Replace every non-overlapping occurrence of
            # this pair with its merged token.
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            self.merges.append(best_pair)

        # Build the final vocabulary from the original
        # characters and all tokens created by our merges.
        vocabulary = set(text)

        for left, right in self.merges:
            vocabulary.add(left + right)

        # Sorting makes token IDs deterministic.
        self.vocab = {
            token: idx
            for idx, token in enumerate(sorted(vocabulary))
        }

        # Convert IDs back to token strings.
        self.inverse_vocab = {
            idx: token
            for token, idx in self.vocab.items()
        }

    def encode(self, text):
        """Convert text into a list of token IDs."""

        tokens = list(text)

        # Apply merges in exactly the order learned during
        # training.
        for left, right in self.merges:
            merged_token = left + right
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # Reject characters that were not in the training
        # vocabulary rather than silently losing information.
        unknown = [token for token in tokens if token not in self.vocab]

        if unknown:
            raise ValueError(
                f"Unknown token(s): {unknown[:5]!r}. "
                "Train on more representative text."
            )

        return [self.vocab[token] for token in tokens]

    def decode(self, ids):
        """Convert token IDs back into text."""

        try:
            return "".join(self.inverse_vocab[idx] for idx in ids)
        except KeyError as exc:
            raise ValueError(
                f"Unknown token ID: {exc.args[0]}"
            ) from exc

    def save(self, path):
        """Save the vocabulary and merge rules as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "merges": [list(pair) for pair in self.merges],
            "vocab": self.vocab,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path):
        """Load a previously saved tokenizer."""

        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        self.merges = [tuple(pair) for pair in data["merges"]]
        self.vocab = data["vocab"]
        self.inverse_vocab = {
            int(idx): token for token, idx in self.vocab.items()
        }
