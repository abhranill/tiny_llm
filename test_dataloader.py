
from torch.utils.data import DataLoader

from utils.dataset import load_corpus, prepare_datasets

text = load_corpus("data/raw/tiny_corpus.txt")

tokenizer, train_dataset, val_dataset = prepare_datasets(
    text,
    sequence_length=4,
    train_ratio=0.8,
    num_merges=25,
)

loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
)

x, y = next(iter(loader))

print("Input batch shape:", tuple(x.shape))
print("Target batch shape:", tuple(y.shape))
print("Input dtype:", x.dtype)
print("Target dtype:", y.dtype)

assert x.shape == (8, 4)
assert y.shape == (8, 4)
assert x.dtype == y.dtype

print("DataLoader test: PASSED")
