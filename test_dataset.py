import torch
from torch.utils.data import DataLoader

from utils.dataset import load_corpus, prepare_datasets


# 1. Load the text file.
corpus_path = "data/raw/tiny_corpus.txt"
text = load_corpus(corpus_path)

print("Corpus characters:", len(text))

# 2. Tokenize and split the data.
tokenizer, train_dataset, val_dataset = prepare_datasets(
    text,
    sequence_length=4,
    train_ratio=0.8,
    num_merges=25,
)

print("Vocabulary size:", len(tokenizer.vocab))
print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))

# 3. Inspect a single training example.
x, y = train_dataset[0]

print("\nFirst input IDs:", x.tolist())
print("First target IDs:", y.tolist())

print("Decoded input:", tokenizer.decode(x.tolist()))
print("Decoded target:", tokenizer.decode(y.tolist()))

# 4. Verify that the target is shifted by one token.
assert len(x) == 4
assert len(y) == 4
assert x.tolist()[1:] == y.tolist()[:-1]
print("Shifted-target test: PASSED")
