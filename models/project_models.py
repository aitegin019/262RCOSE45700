"""Reusable models for the Furiosa Eager/Compile comparison.

Importing this module does not run experiments or access the NPU.
ChunkedConv2d is adapted from the Furiosa mentor's suggestion.
"""
import copy
import torch
import torchvision.models as models
from torch.nn.utils.fusion import fuse_conv_bn_eval


class SmallCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ChunkedConv2d(torch.nn.Module):

    def __init__(self, conv, chunk=128):
        super().__init__()
        if conv.groups != 1:
            raise ValueError('This workaround currently supports groups=1 only.')
        self.convs = torch.nn.ModuleList()
        for start in range(0, conv.out_channels, chunk):
            end = min(start + chunk, conv.out_channels)
            part = torch.nn.Conv2d(in_channels=conv.in_channels, out_channels=end - start, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=1, bias=conv.bias is not None, padding_mode=conv.padding_mode, device=conv.weight.device, dtype=conv.weight.dtype)
            with torch.no_grad():
                part.weight.copy_(conv.weight[start:end])
                if conv.bias is not None:
                    part.bias.copy_(conv.bias[start:end])
            self.convs.append(part)

    def forward(self, x):
        return torch.cat([part(x) for part in self.convs], dim=1)

def fuse_batchnorm(module):
    for child in module.children():
        fuse_batchnorm(child)
    for conv_name, bn_name in [('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3')]:
        if hasattr(module, conv_name) and hasattr(module, bn_name):
            conv = getattr(module, conv_name)
            bn = getattr(module, bn_name)
            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                fused = fuse_conv_bn_eval(conv, bn)
                setattr(module, conv_name, fused)
                setattr(module, bn_name, torch.nn.Identity())
    if isinstance(module, torch.nn.Sequential):
        for i in range(len(module) - 1):
            if isinstance(module[i], torch.nn.Conv2d) and isinstance(module[i + 1], torch.nn.BatchNorm2d):
                module[i] = fuse_conv_bn_eval(module[i], module[i + 1])
                module[i + 1] = torch.nn.Identity()

def replace_large_convs(module, prefix=''):
    count = 0
    for name, child in list(module.named_children()):
        path = f'{prefix}.{name}' if prefix else name
        if isinstance(child, torch.nn.Conv2d) and child.out_channels >= 512:
            if child.groups != 1:
                raise ValueError(f'Grouped convolution unsupported: {path}')
            setattr(module, name, ChunkedConv2d(child, chunk=128))
            print(f'Chunked: {path}', flush=True)
            count += 1
        else:
            count += replace_large_convs(child, path)
    return count


MODEL_SPECS = {
    "smallcnn": {
        "input_shape": (3, 64, 64),
        "validated_batches": (1, 4, 8, 16),
    },
    "resnet18_chunked": {
        "input_shape": (3, 224, 224),
        "validated_batches": (1,),
    },
}


def build_model_pair(name, seed=42):
    """Return independent CPU BF16 models: reference and NPU candidate.

    Both use the same initial weights.
    The ResNet reference is BatchNorm-fused but not chunked.
    The ResNet candidate is BatchNorm-fused and chunked.
    No pretrained weights are loaded.
    """
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}; choose {tuple(MODEL_SPECS)}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        if name == "smallcnn":
            reference = SmallCNN().eval()
        else:
            reference = models.resnet18(weights=None).eval()
            fuse_batchnorm(reference)

        reference = reference.to(dtype=torch.bfloat16)
        candidate = copy.deepcopy(reference)

        if name == "resnet18_chunked":
            count = replace_large_convs(candidate)
            if count != 5:
                raise RuntimeError(f"Expected 5 replacements, got {count}")

    return reference.eval(), candidate.eval()


def make_input(name, batch=1, seed=43):
    """Return a reproducible CPU BF16 input; transfer separately."""
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {name}")
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")

    shape = MODEL_SPECS[name]["input_shape"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.randn(batch, *shape, dtype=torch.bfloat16)
