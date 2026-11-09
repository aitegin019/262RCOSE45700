import torch

from torch.nn.utils.fusion import fuse_conv_bn_eval

# ChunkedConv2d adapted from the Furiosa mentor's suggestion.

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
