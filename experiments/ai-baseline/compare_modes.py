import torch
import torch.nn as nn
import furiosa.torch

torch.manual_seed(42)

device = torch.device("furiosa:0")

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).eval().to(device=device, dtype=torch.bfloat16)

x = torch.randn(
    1, 128,
    dtype=torch.bfloat16,
    device=device
)

# Eager Mode
with torch.inference_mode():
    eager_output = model(x)

print("Eager Mode successful!")
print("Eager output shape:", eager_output.shape)

# Compile Mode using Furiosa backend
compiled_model = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

with torch.inference_mode():
    compiled_output = compiled_model(x)

print("Compile Mode successful!")
print("Compile output shape:", compiled_output.shape)

# Compare outputs
difference = (
    eager_output.float().cpu()
    - compiled_output.float().cpu()
).abs()

print("Maximum absolute difference:", difference.max().item())

print(
    "Outputs close:",
    torch.allclose(
        eager_output.float().cpu(),
        compiled_output.float().cpu(),
        rtol=1e-2,
        atol=1e-2
    )
)
