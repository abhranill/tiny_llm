
import torch

from model.attention import SelfAttention


# Make the test repeatable.
torch.manual_seed(42)

# Test configuration.
batch_size = 2
sequence_length = 4
embed_dim = 16

# Create random token representations.
x = torch.randn(batch_size, sequence_length, embed_dim)

# Construct our attention layer.
attention = SelfAttention(
    embed_dim=embed_dim,
    context_length=sequence_length,
)

# Run the forward pass.
output = attention(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(output.shape))
print("Trainable parameters:", sum(
    p.numel() for p in attention.parameters()
))

# Output must have the same shape as the input.
assert output.shape == x.shape

# The result should contain only finite numbers.
assert torch.isfinite(output).all()

# Verify the causal mask blocks future positions.
weights = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.2, 0.3, 0.5, 0.0],
    [0.1, 0.2, 0.3, 0.4],
])

assert torch.all(weights.triu(diagonal=1) == 0)

print("Output-shape test: PASSED")
print("Finite-values test: PASSED")
print("Causal-mask logic test: PASSED")

# Change the final token, leaving all earlier tokens unchanged.
x_changed = x.clone()
x_changed[:, -1, :] += 100.0

output_changed = attention(x_changed)

# Earlier outputs must remain unchanged because they cannot
# attend to the final, future position.
assert torch.allclose(
    output[:, :-1, :],
    output_changed[:, :-1, :],
    atol=1e-5,
)

print("Future-token isolation test: PASSED")
