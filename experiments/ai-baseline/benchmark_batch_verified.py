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


import time

compiled = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

def benchmark(model_fn, x, warmup=10, runs=50):
    times = []

    with torch.inference_mode():
        for _ in range(warmup):
            model_fn(x).detach().cpu()

        for _ in range(runs):
            start = time.perf_counter()
            model_fn(x).detach().cpu()
            times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)

print("Batch,Eager_ms,Compile_ms,Speedup", flush=True)

for batch in [1, 4, 8, 16]:
    x = torch.randn(
        batch, 3, 64, 64,
        device=device,
        dtype=torch.bfloat16
    )


    with torch.inference_mode():
        eager_output = model(x).detach().cpu().float()
        compiled_output = compiled(x).detach().cpu().float()

    max_diff = (eager_output - compiled_output).abs().max().item()
    is_close = torch.allclose(
        eager_output, compiled_output,
        rtol=1e-2, atol=1e-2
    )

    print(
        f"Batch {batch}: max_diff={max_diff:.6f}, "
        f"outputs_close={is_close}",
        flush=True
    )

    eager_ms = benchmark(model, x)
    compile_ms = benchmark(compiled, x)

    print(
        f"{batch},{eager_ms:.3f},{compile_ms:.3f},"
        f"{eager_ms / compile_ms:.2f}",
        flush=True
    )
