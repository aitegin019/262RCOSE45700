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

print("Model loaded successfully!")

x = torch.randn(
    1, 3, 224, 224,
    dtype=torch.bfloat16,
    device=device
)

print("Running Eager Mode inference...")

with torch.inference_mode():
    output = model(x)

print("Inference successful!")
print("Output shape:", output.shape)
print("Output device:", output.device)
print("Output dtype:", output.dtype)
print("Output contains NaN:", torch.isnan(output.cpu()).any().item())
