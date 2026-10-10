
"""A small educational Byte Pair Encoding tokenizer."""

from collections import Counter
import json
from pathlib import Path


class BPETokenizer:
    """Learn frequent adjacent-token merges from text."""

    def __init__(self):
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}
        self.unk_token = "<UNK>"

    def train(self, text, num_merges=100):
        if not text:
            raise ValueError("Training text cannot be empty.")
        if num_merges < 0:
            raise ValueError("num_merges cannot be negative.")

        # Reset state so repeated training starts from scratch.
        self.merges = []
        tokens = list(text)

        for _ in range(num_merges):
            pair_counts = Counter(zip(tokens, tokens[1:]))

            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < 2:
                break

            left, right = best_pair
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
            self.merges.append(best_pair)

        # Include original characters, merged tokens and UNK.
        vocabulary = set(text)
        for left, right in self.merges:
            vocabulary.add(left + right)

        vocabulary.add(self.unk_token)

        self.vocab = {
            token: idx
            for idx, token in enumerate(sorted(vocabulary))
        }

        self.inverse_vocab = {
            idx: token for token, idx in self.vocab.items()
        }

    def encode(self, text):
        tokens = list(text)

        # Apply learned merges in the same order used during training.
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

        # Map tokens absent from the vocabulary to the UNK token.
        unk_id = self.vocab[self.unk_token]
        return [
            self.vocab.get(token, unk_id)
            for token in tokens
        ]

    def decode(self, ids):
        try:
            return "".join(self.inverse_vocab[int(idx)] for idx in ids)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid token ID: {exc}"
            ) from exc

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "merges": [list(pair) for pair in self.merges],
            "vocab": self.vocab,
            "unk_token": self.unk_token,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path):
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        self.merges = [
            tuple(pair) for pair in data["merges"]
        ]
        self.vocab = {
            token: int(idx)
            for token, idx in data["vocab"].items()
        }
        self.unk_token = data.get("unk_token", "<UNK>")

        # Support tokenizer files saved before UNK was introduced.
        if self.unk_token not in self.vocab:
            self.vocab[self.unk_token] = max(
                self.vocab.values(), default=-1
            ) + 1

        self.inverse_vocab = {
            idx: token for token, idx in self.vocab.items()
        }
