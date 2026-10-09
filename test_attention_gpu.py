
import torch

from model.attention import SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = SelfAttention(
    embed_dim=64,
    context_length=128,
).to(device)

x = torch.randn(2, 128, 64, device=device)

output = attention(x)

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(output.shape))

assert output.shape == x.shape
assert torch.isfinite(output).all()

print("GPU attention test: PASSED")
