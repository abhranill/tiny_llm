
from pathlib import Path
from tokenizer.bpe import BPETokenizer

tokenizer = BPETokenizer()

training_text = "hello world hello machine learning"
tokenizer.train(training_text, num_merges=20)

# 1. Known text should round-trip exactly.
known_text = "hello world"
known_ids = tokenizer.encode(known_text)
assert tokenizer.decode(known_ids) == known_text
print("Known-text round trip: PASSED")

# 2. Unseen characters should not crash encoding.
unseen_text = "hello 🐍"
unseen_ids = tokenizer.encode(unseen_text)
assert len(unseen_ids) > 0
assert tokenizer.vocab[tokenizer.unk_token] in unseen_ids
print("Unknown-character fallback: PASSED")

# 3. Save and load should preserve the vocabulary and merges.
test_path = Path("tokenizer/test_tokenizer.json")
tokenizer.save(test_path)

loaded = BPETokenizer()
loaded.load(test_path)

assert loaded.vocab == tokenizer.vocab
assert loaded.merges == tokenizer.merges
assert loaded.encode(known_text) == known_ids
print("Save/load test: PASSED")

test_path.unlink()
print("Tokenizer robustness tests: PASSED")
