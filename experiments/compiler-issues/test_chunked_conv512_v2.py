import torch
import furiosa.torch
from furiosa.torch import config


# Adapted from the Furiosa mentor's suggested workaround.
class ChunkedConv2d(torch.nn.Module):
    def __init__(self, conv, chunk=128):
        super().__init__()
        if conv.groups != 1:
            raise ValueError("This workaround currently supports groups=1 only.")

        self.convs = torch.nn.ModuleList()

        for start in range(0, conv.out_channels, chunk):
            end = min(start + chunk, conv.out_channels)

            part = torch.nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=end - start,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=1,
                bias=conv.bias is not None,
                padding_mode=conv.padding_mode,
                device=conv.weight.device,
                dtype=conv.weight.dtype,
            )

            with torch.no_grad():
                part.weight.copy_(conv.weight[start:end])
                if conv.bias is not None:
                    part.bias.copy_(conv.bias[start:end])

            self.convs.append(part)

    def forward(self, x):
        return torch.cat([part(x) for part in self.convs], dim=1)


def check(label, actual, expected):
    actual = actual.float()
    expected = expected.float()

    max_diff = (actual - expected).abs().max().item()
    finite = bool(torch.isfinite(actual).all())
    close = torch.allclose(
        actual, expected, rtol=1e-2, atol=1e-2
    )

    print(
        f"{label}: max_diff={max_diff:.8f}, "
        f"finite={finite}, outputs_close={close}",
        flush=True,
    )

    if not finite or not close:
        raise AssertionError(f"Correctness check failed: {label}")


torch.manual_seed(42)
device = torch.device("furiosa:0")

original = torch.nn.Conv2d(
    512, 512, kernel_size=3, padding=1
).eval().to(dtype=torch.bfloat16)

# Copy the original weights into four output-channel chunks.
chunked = ChunkedConv2d(original, chunk=128).eval()

# Keep the original model on CPU as the reference.
chunked = chunked.to(device)

compiled = torch.compile(
    chunked,
    backend=furiosa.torch.backend,
    mode="default",
)

print("Testing 512-channel convolution with chunks of 128.", flush=True)

# Disable CPU fallback to expose unsupported operations.
with config.cpu_fallback(False):
    with torch.inference_mode():
        for seed in [42, 43, 44]:
            torch.manual_seed(seed)
            x = torch.randn(
                1, 512, 4, 4, dtype=torch.bfloat16
            ).to(device)

            print(f"\nInput seed: {seed}", flush=True)

            reference = original(x.cpu()).cpu()
            print("Original CPU reference completed.", flush=True)

            eager_result = chunked(x).cpu()
            print("Chunked Eager completed.", flush=True)

            compile_result = compiled(x).cpu()
            print("Chunked Compile completed.", flush=True)

            check("CPU reference vs chunked NPU Eager", eager_result, reference)
            check("CPU reference vs chunked NPU Compile", compile_result, reference)
            check("Chunked Eager vs Compile", compile_result, eager_result)

print("\nSUCCESS: all three inputs passed.", flush=True)
