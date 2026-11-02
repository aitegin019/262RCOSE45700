import torch
import torchvision.models as models
import furiosa.torch

torch.manual_seed(42)

device = torch.device("furiosa:0")

print("Loading ResNet-18...")

model = models.resnet18(weights=None)
model = model.eval().to(
    device=device,
    dtype=torch.bfloat16
)

x = torch.randn(
    1, 3, 224, 224,
    device=device,
    dtype=torch.bfloat16
)

# Eager Mode
print("\nRunning Eager Mode...")

with torch.no_grad():
    eager_output = model(x)

print("Eager Mode successful!")

# Compile Mode
print("\nPreparing Compile Mode...")

compiled_model = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

print("Running Compile Mode...")

with torch.no_grad():
    compiled_output = compiled_model(x)

print("Compile Mode successful!")

# Compare outputs on CPU
eager_cpu = eager_output.detach().cpu().float()
compiled_cpu = compiled_output.detach().cpu().float()

max_diff = torch.max(
    torch.abs(eager_cpu - compiled_cpu)
).item()

outputs_close = torch.allclose(
    eager_cpu,
    compiled_cpu,
    rtol=1e-2,
    atol=1e-2
)

print("\n===== RESULTS =====")
print("Eager output shape:", eager_output.shape)
print("Compile output shape:", compiled_output.shape)
print("Maximum absolute difference:", max_diff)
print("Outputs close:", outputs_close)
print("===================")
