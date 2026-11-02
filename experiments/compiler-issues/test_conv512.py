import torch
import furiosa.torch

device = torch.device("furiosa:0")

model = torch.nn.Conv2d(
    in_channels=512,
    out_channels=512,
    kernel_size=3,
    padding=1
).eval().to(device=device, dtype=torch.bfloat16)

x = torch.randn(
    1, 512, 4, 4,
    device=device,
    dtype=torch.bfloat16
)

print("Running Eager Mode...", flush=True)

with torch.no_grad():
    eager = model(x)

print("Eager Mode successful!", flush=True)

print("Running Compile Mode...", flush=True)

compiled = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

with torch.no_grad():
    result = compiled(x)

print("Compile Mode successful!", flush=True)

a = eager.detach().cpu().float()
b = result.detach().cpu().float()

print("Max difference:", (a - b).abs().max().item())
print("Outputs close:", torch.allclose(a, b, rtol=1e-2, atol=1e-2))
