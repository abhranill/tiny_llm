
import torch
from model.attention import MultiHeadSelfAttention

torch.manual_seed(42)

# Create a 4-head attention layer.
attention = MultiHeadSelfAttention(
    embed_dim=64,
    num_heads=4,
    context_length=16,
)
attention.eval()

# Two examples, each containing eight tokens.
x = torch.randn(2, 8, 64)

# Run the attention layer.
output = attention(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(output.shape))
print("All outputs finite:", torch.isfinite(output).all().item())

assert output.shape == x.shape
assert torch.isfinite(output).all()

# Change the final token.
# Earlier output positions should remain unchanged.
x_changed = x.clone()
x_changed[:, -1, :] += 100.0

output_changed = attention(x_changed)

assert torch.allclose(
    output[:, :-1, :],
    output_changed[:, :-1, :],
    atol=1e-5,
)

print("Output shape test: PASSED")
print("Finite values test: PASSED")
print("Causal isolation test: PASSED")
