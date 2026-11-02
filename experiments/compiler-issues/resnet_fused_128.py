import torch
import torchvision.models as models
import furiosa.torch
from torch.nn.utils.fusion import fuse_conv_bn_eval

def fuse_batchnorm(module):
    for child in module.children():
        fuse_batchnorm(child)

    for conv_name, bn_name in [
        ("conv1", "bn1"),
        ("conv2", "bn2"),
        ("conv3", "bn3")
    ]:
        if hasattr(module, conv_name) and hasattr(module, bn_name):
            conv = getattr(module, conv_name)
            bn = getattr(module, bn_name)

            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                fused = fuse_conv_bn_eval(conv, bn)
                setattr(module, conv_name, fused)
                setattr(module, bn_name, torch.nn.Identity())

    if isinstance(module, torch.nn.Sequential):
        for i in range(len(module) - 1):
            if isinstance(module[i], torch.nn.Conv2d) and isinstance(module[i+1], torch.nn.BatchNorm2d):
                module[i] = fuse_conv_bn_eval(module[i], module[i+1])
                module[i+1] = torch.nn.Identity()

torch.manual_seed(42)

print("Loading ResNet-18...")
model = models.resnet18(weights=None).eval()

print("Fusing BatchNorm layers...")
fuse_batchnorm(model)

device = torch.device("furiosa:0")
model = model.to(device=device, dtype=torch.bfloat16)

x = torch.randn(
    1, 3, 128, 128,
    device=device,
    dtype=torch.bfloat16
)

print("Running Eager Mode...")
with torch.no_grad():
    eager_output = model(x)
print("Eager Mode successful!")

print("Preparing Compile Mode...")
compiled_model = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

print("Running Compile Mode...")
with torch.no_grad():
    compiled_output = compiled_model(x)
print("Compile Mode successful!")

eager_cpu = eager_output.detach().cpu().float()
compiled_cpu = compiled_output.detach().cpu().float()

print("\n===== RESULTS =====")
print("Max difference:", (eager_cpu - compiled_cpu).abs().max().item())
print("Outputs close:", torch.allclose(
    eager_cpu, compiled_cpu,
    rtol=1e-2, atol=1e-2
))
