
import torch

from model.transformer import TinyGPT

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our current toy tokenizer has a vocabulary of 50 tokens.
model = TinyGPT(
    vocab_size=50,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    context_length=128,
    dropout=0.0,
).to(device)

# Two examples, eight tokens per example.
input_ids = torch.randint(0, 50, (2, 8), device=device)

logits = model(input_ids)

print("Device:", device)
print("Input shape:", tuple(input_ids.shape))
print("Logits shape:", tuple(logits.shape))
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

assert logits.shape == (2, 8, 50)
assert torch.isfinite(logits).all()

# Confirm gradients flow through the complete model.
loss = logits.square().mean()
loss.backward()

assert all(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
)

# Check the context-length limit.
try:
    model(torch.randint(0, 50, (1, 129), device=device))
    raise AssertionError("Expected a context-length error.")
except ValueError:
    pass

print("Output-shape test: PASSED")
print("Finite-logits test: PASSED")
print("Gradient-flow test: PASSED")
print("Context-length test: PASSED")
print("TinyGPT test: PASSED")
