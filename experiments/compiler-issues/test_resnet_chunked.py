import copy
import torch
import torchvision.models as models
import furiosa.torch
from furiosa.torch import config
from resnet_workaround_helpers import ChunkedConv2d, fuse_batchnorm


def replace_large_convs(module, prefix=""):
    count = 0
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Conv2d) and child.out_channels >= 512:
            if child.groups != 1:
                raise ValueError(f"Grouped convolution unsupported: {path}")
            setattr(module, name, ChunkedConv2d(child, chunk=128))
            print(f"Chunked: {path}", flush=True)
            count += 1
        else:
            count += replace_large_convs(child, path)
    return count


def check(label, actual, expected):
    actual = actual.float()
    expected = expected.float()
    finite = bool(
        torch.isfinite(actual).all()
        and torch.isfinite(expected).all()
    )
    max_diff = (actual - expected).abs().max().item()
    close = torch.allclose(
        actual, expected, rtol=1e-2, atol=1e-2
    )
    print(
        f"{label}: max_diff={max_diff:.8f}, "
        f"finite={finite}, outputs_close={close}",
        flush=True,
    )
    if not finite or not close:
        raise AssertionError(label)


torch.manual_seed(42)

print("Preparing ResNet-18 with random weights.", flush=True)
reference = models.resnet18(weights=None).eval()
fuse_batchnorm(reference)
reference = reference.to(dtype=torch.bfloat16)

model = copy.deepcopy(reference)
count = replace_large_convs(model)
print(f"Replaced {count} convolutions.", flush=True)

torch.manual_seed(43)
x_cpu = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

with torch.inference_mode():
    expected = reference(x_cpu)
    chunked_cpu = model(x_cpu)
    check("Original fused CPU vs chunked CPU", chunked_cpu, expected)

device = torch.device("furiosa:0")
model = model.eval().to(device)
x_npu = x_cpu.to(device)

with config.cpu_fallback(False):
    with torch.inference_mode():
        print("Running chunked ResNet NPU Eager...", flush=True)
        eager = model(x_npu).cpu()
        check("CPU reference vs NPU Eager", eager, expected)

        print("Compiling chunked ResNet...", flush=True)
        compiled = torch.compile(
            model,
            backend=furiosa.torch.backend,
            mode="default",
        )

        result = compiled(x_npu).cpu()
        print("Chunked ResNet Compile completed.", flush=True)

        check("CPU reference vs NPU Compile", result, expected)
        check("NPU Eager vs Compile", result, eager)

print(
    "\nSUCCESS: chunked ResNet-18 passed for this input.",
    flush=True,
)
