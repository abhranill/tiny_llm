
import torch
from model.transformer import TinyGPT

model = TinyGPT(
    vocab_size=71,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    context_length=128,
)

input_ids = torch.tensor([[1, 2, 3]])

# Test valid generation.
output = model.generate(
    input_ids,
    max_new_tokens=2,
    temperature=0.8,
    top_k=10,
)

assert output.shape == (1, 5)
print("Valid generation: PASSED")

# Test invalid settings.
checks = [
    ("zero temperature", {"temperature": 0}),
    ("zero top_k", {"top_k": 0}),
    ("negative token count", {"max_new_tokens": -1}),
]

for name, kwargs in checks:
    try:
        model.generate(input_ids, **kwargs)
    except ValueError:
        print(f"{name}: PASSED")
    else:
        raise AssertionError(f"{name} should have raised ValueError")

print("All generation validation tests passed!")
