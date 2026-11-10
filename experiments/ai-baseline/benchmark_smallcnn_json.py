import argparse
import json
from pathlib import Path

from furiosa.torch import config

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

with config.cpu_fallback(False):
    import torch
    import furiosa.torch

    import sys

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

    from models.project_models import build_model_pair, make_input

    reference, model = build_model_pair("smallcnn", seed=42)
    device = torch.device("furiosa:0")
    model = model.to(device)

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

        return times

    print("Batch,Eager_ms,Compile_ms,Speedup", flush=True)

    results = []

    for batch in [1, 4, 8, 16]:
        x_cpu = make_input("smallcnn", batch=batch, seed=43)
        with torch.inference_mode():
            reference_output = reference(x_cpu).float()
        x = x_cpu.to(device)


        with torch.inference_mode():
            start = time.perf_counter()
            eager_output = model(x).detach().cpu()
            eager_first_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            compiled_output = compiled(x).detach().cpu()
            compile_first_ms = (time.perf_counter() - start) * 1000

            eager_output = eager_output.float()
            compiled_output = compiled_output.float()

        print(
            f"Batch {batch} first call: Eager={eager_first_ms:.3f} ms, "
            f"Compile={compile_first_ms:.3f} ms",
            flush=True
        )

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

        eager_matches_cpu = torch.allclose(
            eager_output, reference_output, rtol=1e-2, atol=1e-2
        )
        compile_matches_cpu = torch.allclose(
            compiled_output, reference_output, rtol=1e-2, atol=1e-2
        )
        print(
            f"CPU checks: Eager={eager_matches_cpu}, "
            f"Compile={compile_matches_cpu}",
            flush=True,
        )

        if not (is_close and eager_matches_cpu and compile_matches_cpu):
            results.append({
                "batch_size": batch,
                "status": "correctness_failed",
                "outputs_close": is_close,
                "eager_matches_cpu": eager_matches_cpu,
                "compile_matches_cpu": compile_matches_cpu,
                "max_abs_diff": max_diff,
            })
            continue

        eager_samples = benchmark(model, x)
        compile_samples = benchmark(compiled, x)
        eager_ms = sum(eager_samples) / len(eager_samples)
        compile_ms = sum(compile_samples) / len(compile_samples)

        print(
            f"{batch},{eager_ms:.3f},{compile_ms:.3f},"
            f"{eager_ms / compile_ms:.2f}",
            flush=True
        )
        results.append({
            "batch_size": batch,
            "status": "passed",
            "eager_matches_cpu": eager_matches_cpu,
            "compile_matches_cpu": compile_matches_cpu,
            "eager_first_call_ms": eager_first_ms,
            "compile_first_call_ms": compile_first_ms,
            "eager_warm_samples_ms": eager_samples,
            "compile_warm_samples_ms": compile_samples,
            "eager_warm_mean_ms": eager_ms,
            "compile_warm_mean_ms": compile_ms,
            "compile_warm_speedup": eager_ms / compile_ms,
            "max_abs_diff": max_diff,
            "outputs_close": is_close,
        })

import platform
from importlib.metadata import version

report = {
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "furiosa_torch": version("furiosa-torch"),
    },
    "model": "SmallCNN",
    "device": "furiosa:0",
    "dtype": "bfloat16",
    "seed": 42,
    "input_seed": 43,
    "cpu_fallback": False,
    "warmup_iterations": 10,
    "timed_iterations": 50,
    "rtol": 1e-2,
    "atol": 1e-2,
    "timing_scope": "Input already on NPU; output copy to CPU included",
    "first_call_protocol": "Per batch shape, same process, Eager first, cache state uncontrolled",
    "results": results,
}
with args.output.open("x") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print(f"Saved {len(results)} results to {args.output}")
if any(row["status"] != "passed" for row in results):
    raise SystemExit("One or more batches failed the correctness check.")
