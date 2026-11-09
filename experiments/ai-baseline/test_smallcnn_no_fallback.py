import copy
import csv
import torch
import furiosa.torch
from furiosa.torch import config

class SmallCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

torch.manual_seed(42)
reference = SmallCNN().eval().to(dtype=torch.bfloat16)
model = copy.deepcopy(reference).to("furiosa:0")

compiled = torch.compile(
    model, backend=furiosa.torch.backend, mode="default"
)

failures = 0

with open("smallcnn_no_fallback_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "seed", "batch", "comparison", "max_abs_diff",
        "finite", "outputs_close", "rtol", "atol"
    ])
    file.flush()

    with config.cpu_fallback(False):
        with torch.inference_mode():
            for seed in [43, 44, 45, 46, 47]:
                print(f"\nTesting seed={seed}, batch=1", flush=True)
                torch.manual_seed(seed)
                x_cpu = torch.randn(
                    1, 3, 64, 64, dtype=torch.bfloat16
                )
                expected = reference(x_cpu).float()
                x_npu = x_cpu.to("furiosa:0")

                eager = model(x_npu).cpu().float()
                print("NPU Eager completed.", flush=True)

                result = compiled(x_npu).cpu().float()
                print("NPU Compile completed.", flush=True)

                for label, actual, target in [
                    ("CPU_vs_Eager", eager, expected),
                    ("CPU_vs_Compile", result, expected),
                    ("Eager_vs_Compile", result, eager),
                ]:
                    finite = bool(
                        torch.isfinite(actual).all()
                        and torch.isfinite(target).all()
                    )
                    difference = (actual - target).abs().max().item()
                    close = bool(torch.allclose(
                        actual, target, rtol=1e-2, atol=1e-2
                    ))
                    failures += int(not (finite and close))

                    writer.writerow([
                        seed, 1, label, difference,
                        finite, close, 1e-2, 1e-2
                    ])
                    file.flush()
                    print(
                        f"{label}: max_diff={difference:.8f}, "
                        f"finite={finite}, outputs_close={close}",
                        flush=True,
                    )

print(f"\nFinished: {failures} failed comparisons out of 15.", flush=True)
if failures:
    raise SystemExit(1)
print("SUCCESS: all five inputs passed with CPU fallback disabled.", flush=True)
