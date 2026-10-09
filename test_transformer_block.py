
import torch

from model.transformer import TransformerBlock

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block = TransformerBlock(
    embed_dim=64,
    num_heads=4,
    context_length=16,
    dropout=0.0,
).to(device)

x = torch.randn(2, 8, 64, device=device)
output = block(x)

print("Device:", device)
print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(output.shape))
print("All outputs finite:", torch.isfinite(output).all().item())

assert output.shape == x.shape
assert torch.isfinite(output).all()

# Check that gradients can flow through the block.
loss = output.square().mean()
loss.backward()

has_gradients = all(
    parameter.grad is not None
    for parameter in block.parameters()
    if parameter.requires_grad
)

assert has_gradients
print("Gradient-flow test: PASSED")
print("Transformer block test: PASSED")
