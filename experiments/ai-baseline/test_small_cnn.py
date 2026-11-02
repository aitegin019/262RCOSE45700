import torch
import furiosa.torch

class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

torch.manual_seed(42)
device = torch.device("furiosa:0")

model = SmallCNN().eval().to(
    device=device, dtype=torch.bfloat16
)

x = torch.randn(
    1, 3, 64, 64,
    device=device,
    dtype=torch.bfloat16
)

print("Running Eager Mode...", flush=True)

with torch.no_grad():
    eager = model(x)

print("Eager Mode successful!", flush=True)

compiled = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

print("Running Compile Mode...", flush=True)

with torch.no_grad():
    result = compiled(x)

print("Compile Mode successful!", flush=True)

a = eager.detach().cpu().float()
b = result.detach().cpu().float()

print("Max difference:", (a - b).abs().max().item())
print("Outputs close:", torch.allclose(
    a, b, rtol=1e-2, atol=1e-2
))
