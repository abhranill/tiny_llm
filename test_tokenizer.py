
from pathlib import Path

from tokenizer.bpe import BPETokenizer


# Small sample text for our first experiment.
training_text = (
    "hello world hello world "
    "hello machine learning "
    "hello world machine learning"
)

# Create a tokenizer.
tokenizer = BPETokenizer()

# Learn 30 merges from the sample text.
tokenizer.train(training_text, num_merges=30)

# Convert text to token IDs.
encoded = tokenizer.encode("hello world")

# Convert the IDs back to text.
decoded = tokenizer.decode(encoded)

print("Training text:", training_text)
print("Vocabulary size:", len(tokenizer.vocab))
print("Number of merges:", len(tokenizer.merges))
print("Encoded IDs:", encoded)
print("Decoded text:", decoded)

# Encoding and decoding must preserve the text.
assert decoded == "hello world"

# Save the tokenizer.
save_path = Path("tokenizer/tokenizer.json")
tokenizer.save(save_path)

# Load it into a fresh tokenizer.
loaded_tokenizer = BPETokenizer()
loaded_tokenizer.load(save_path)

# The saved tokenizer must behave identically.
assert loaded_tokenizer.decode(
    loaded_tokenizer.encode("hello world")
) == "hello world"

assert loaded_tokenizer.vocab == tokenizer.vocab
assert loaded_tokenizer.merges == tokenizer.merges

print("Round-trip test: PASSED")
print("Save/load test: PASSED")
