import torch
import furiosa.torch
from furiosa.torch import config


class AttentionCore(torch.nn.Module):
    """Unmasked scaled dot-product attention; no dropout or projections."""

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (q.shape[-1] ** -0.5)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v)


def check(name, actual, expected):
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
        f"{name}: max_diff={max_diff:.8f}, "
        f"finite={finite}, outputs_close={close}",
        flush=True,
    )
    if not (finite and close):
        raise AssertionError(f"{name} failed")


def main():
    # Shape: batch, heads, sequence length, head dimension.
    shape = (1, 4, 64, 64)
    cpu_model = AttentionCore().eval()
    npu_model = AttentionCore().eval().to("furiosa:0")

    with config.cpu_fallback(False), torch.inference_mode():
        compiled = torch.compile(
            npu_model,
            backend=furiosa.torch.backend,
            mode="default",
        )

        for seed in [43, 44, 45, 46, 47]:
            generator = torch.Generator().manual_seed(seed)
            inputs = tuple(
                torch.randn(
                    shape,
                    generator=generator,
                    dtype=torch.bfloat16,
                )
                for _ in range(3)
            )

            print(f"\nTesting seed={seed}, shape={shape}", flush=True)
            expected = cpu_model(*inputs)
            npu_inputs = tuple(x.to("furiosa:0") for x in inputs)

            print("Running NPU Eager...", flush=True)
            eager = npu_model(*npu_inputs).cpu()

            print("Running NPU Compile...", flush=True)
            result = compiled(*npu_inputs).cpu()

            check("CPU vs Eager", eager, expected)
            check("CPU vs Compile", result, expected)
            check("Eager vs Compile", result, eager)

    print(
        "\nSUCCESS: all five attention inputs passed "
        "with CPU fallback disabled.",
        flush=True,
    )


if __name__ == "__main__":
    main()
