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

x = torch.randn(
    1, 3, 64, 64,
    device=device,
    dtype=torch.bfloat16
)


import time

compiled = torch.compile(
    model,
    backend=furiosa.torch.backend,
    mode="default"
)

def benchmark(name, model_fn, x, warmup=10, runs=50):
    print(f"Benchmarking {name}...", flush=True)

    with torch.inference_mode():
        for _ in range(warmup):
            model_fn(x)

        # Synchronize by transferring the final output to CPU
        # before starting the timer.
        model_fn(x).detach().cpu()

        times = []

        for _ in range(runs):
            start = time.perf_counter()
            output = model_fn(x)
            output.detach().cpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    avg = sum(times) / len(times)
    print(f"{name} average latency: {avg:.3f} ms", flush=True)
    return avg

eager_ms = benchmark("Eager", model, x)
compile_ms = benchmark("Compile", compiled, x)

print("\\n===== PERFORMANCE RESULTS =====")
print(f"Eager latency: {eager_ms:.3f} ms")
print(f"Compile latency: {compile_ms:.3f} ms")
print(f"Speedup: {eager_ms / compile_ms:.2f}x")
