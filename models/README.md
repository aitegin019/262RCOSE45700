# Reusable inference models

These models require the provided Furiosa NPU environment.
Importing project_models does not run experiments or access the NPU.

## Available models

- smallcnn: random weights, BF16, input [B, 3, 64, 64].
- resnet18_chunked: random weights, BatchNorm-fused ResNet-18,
  with five convolutions split into 128-output-channel chunks.
  Input: [B, 3, 224, 224].

## Correctness validation

CPU fallback was disabled for the recorded NPU checks.

- SmallCNN: batches 1, 4, 8, 16; five inputs per batch passed.
- Modified ResNet-18: batch 1; five inputs passed.
- ResNet batch 4: NPU Eager fails with:
  "insufficient sequencer entries: required 9".

Working tolerances: rtol=1e-2, atol=1e-2.
These checks establish numerical agreement for tested inputs,
not classification accuracy.

## Verify from the repository root

    python models/verify_project_models.py --model smallcnn
    python models/verify_project_models.py --model resnet18_chunked

## Use in a benchmark runner

For a runner located at the repository root:

    import torch
    import furiosa.torch
    from furiosa.torch import config
    from models.project_models import build_model_pair, make_input

    reference, model = build_model_pair("smallcnn", seed=42)
    x_cpu = make_input("smallcnn", batch=1, seed=43)

    with config.cpu_fallback(False):
        model = model.to("furiosa:0")
        x_npu = x_cpu.to("furiosa:0")
        compiled = torch.compile(
            model,
            backend=furiosa.torch.backend,
            mode="default",
        )

        with torch.inference_mode():
            eager_output = model(x_npu).cpu()
            compiled_output = compiled(x_npu).cpu()

Use the same modified model, weights, and inputs for both modes.
Keep CPU fallback disabled during timed experiments too.

## Benchmarking handoff

The benchmark runner should:
- Verify outputs before measuring performance.
- Separate first-run costs from warmed-up execution.
- State whether input/output transfers are included.
- Save individual measurements and environment information.
- Record unsupported configurations as failures, not speed results.

October 27 timings remain preliminary.
See results/2026-11-09/ for the expanded correctness checks.
