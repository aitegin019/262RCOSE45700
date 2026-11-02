import torch
import torch.nn as nn
import furiosa.torch

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

model = model.eval().to(
    device="furiosa:0",
    dtype=torch.bfloat16
)

x = torch.randn(
    1, 128,
    dtype=torch.bfloat16,
    device="furiosa:0"
)

with torch.inference_mode():
    output = model(x)

print("Output shape:", output.shape)
print("Output device:", output.device)
print("Inference successful!")
