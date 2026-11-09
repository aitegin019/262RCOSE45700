import copy
import csv
import torch
import torchvision.models as models
import furiosa.torch
from furiosa.torch import config
from resnet_workaround_helpers import ChunkedConv2d, fuse_batchnorm

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

torch.manual_seed(42)
reference = models.resnet18(weights=None).eval()
fuse_batchnorm(reference)
reference = reference.to(dtype=torch.bfloat16)

model = copy.deepcopy(reference)
count = replace_large_convs(model)
print(f"Replaced {count} convolutions.", flush=True)

device = torch.device("furiosa:0")
model = model.eval().to(device)

compiled = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default",
)

failures = 0

with open("resnet_multi_input_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "seed", "batch", "comparison", "max_abs_diff",
            "finite", "outputs_close", "rtol", "atol"
        ],
    )
    writer.writeheader()
    file.flush()

    with config.cpu_fallback(False):
        with torch.inference_mode():
            for seed in [43, 44, 45, 46, 47]:
                print(f"\nTesting seed={seed}, batch=1", flush=True)
                torch.manual_seed(seed)
                x_cpu = torch.randn(
                    1, 3, 224, 224, dtype=torch.bfloat16
                )
                expected = reference(x_cpu).float()
                x_npu = x_cpu.to(device)

                eager = model(x_npu).cpu().float()
                print("NPU Eager completed.", flush=True)

                result = compiled(x_npu).cpu().float()
                print("NPU Compile completed.", flush=True)

                comparisons = [
                    ("CPU_vs_Eager", eager, expected),
                    ("CPU_vs_Compile", result, expected),
                    ("Eager_vs_Compile", result, eager),
                ]

                for label, actual, target in comparisons:
                    finite = bool(
                        torch.isfinite(actual).all()
                        and torch.isfinite(target).all()
                    )
                    difference = (actual - target).abs().max().item()
                    close = bool(torch.allclose(
                        actual, target, rtol=1e-2, atol=1e-2
                    ))
                    passed = finite and close
                    failures += int(not passed)

                    writer.writerow({
                        "seed": seed,
                        "batch": 1,
                        "comparison": label,
                        "max_abs_diff": difference,
                        "finite": finite,
                        "outputs_close": close,
                        "rtol": 1e-2,
                        "atol": 1e-2,
                    })
                    file.flush()

                    print(
                        f"{label}: max_diff={difference:.8f}, "
                        f"finite={finite}, outputs_close={close}",
                        flush=True,
                    )

print(f"\nFinished: {failures} failed comparisons out of 15.", flush=True)
if failures:
    raise SystemExit(1)
print("SUCCESS: all five inputs passed.", flush=True)
