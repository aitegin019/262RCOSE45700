import argparse
import torch
import furiosa.torch
from furiosa.torch import config
from project_models import build_model_pair, make_input

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["smallcnn", "resnet18_chunked"],
    required=True,
)
args = parser.parse_args()

reference, candidate = build_model_pair(args.model)
x_cpu = make_input(args.model, batch=1, seed=43)

with torch.inference_mode():
    expected = reference(x_cpu).float()

with config.cpu_fallback(False):
    candidate = candidate.to("furiosa:0")
    x_npu = x_cpu.to("furiosa:0")

    with torch.inference_mode():
        print("Running NPU Eager...", flush=True)
        eager = candidate(x_npu).cpu().float()

        compiled = torch.compile(
            candidate,
            backend=furiosa.torch.backend,
            mode="default",
        )

        print("Running NPU Compile...", flush=True)
        result = compiled(x_npu).cpu().float()

for label, actual, target in [
    ("CPU vs Eager", eager, expected),
    ("CPU vs Compile", result, expected),
    ("Eager vs Compile", result, eager),
]:
    assert torch.isfinite(actual).all(), label
    assert torch.isfinite(target).all(), label
    torch.testing.assert_close(
        actual, target, rtol=1e-2, atol=1e-2
    )
    print(
        f"{label}: PASS, "
        f"max_diff={(actual - target).abs().max().item():.8f}",
        flush=True,
    )

print(f"SUCCESS: {args.model} module verified on NPU.", flush=True)
