
from tokenizer.bpe import BPETokenizer

# Load the tokenizer we trained earlier.
tokenizer = BPETokenizer()
tokenizer.load("tokenizer/tokenizer.json")

# Try several strings that use characters from training.
samples = [
    "hello world",
    "hello machine",
    "machine learning",
    "world",
]

for text in samples:
    ids = tokenizer.encode(text)
    reconstructed = tokenizer.decode(ids)

    print(f"Text:       {text!r}")
    print(f"Token IDs:  {ids}")
    print(f"Num tokens: {len(ids)}")
    print(f"Decoded:    {reconstructed!r}")
    print("-" * 35)
