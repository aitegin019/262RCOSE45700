# SmallCNN JSON benchmark

Run from the repository root in the provided Furiosa NPU environment:

    python experiments/ai-baseline/benchmark_smallcnn_json.py --output smallcnn-results.json

Use a new output filename for each run. The output directory must exist.

The runner uses the shared SmallCNN model with random BF16 weights, model seed 42, and input seed 43. CPU fallback is disabled. Batch sizes are 1, 4, 8, and 16.

Eager and Compile outputs are compared with each other and with the CPU reference using rtol=0.01 and atol=0.01. Failed comparisons produce a failure row and skip warm timing for that batch. The process exits unsuccessfully if any batch fails correctness.

Warm measurements use 10 warm-up calls and 50 individually timed calls per mode. Inputs are already on the NPU. Timing includes the model call and output transfer to CPU.

JSON includes individual timings, averages, speedup, correctness results, and software versions.

First-call measurements use one process, Eager first, and uncontrolled cache state. They are exploratory. Controlled cold-start trials, repeated runs, memory measurements, and profiling remain future work.

Validated on November 10, 2026 with Python 3.12.3, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu, and furiosa-torch 2026.4.0. All four batches passed the correctness checks, with 400 individual warm timings saved.

## Chunked ResNet-18 benchmark

Run from the repository root in the provided Furiosa environment:

    python experiments/ai-baseline/benchmark_smallcnn_json.py --model resnet18_chunked --output resnet18-results.json

The --model option defaults to smallcnn, which runs batches 1, 4, 8, and 16. Selecting resnet18_chunked runs batch 1.

The ResNet candidate uses random BF16 weights, fused BatchNorm, and five convolutions split into 128-output-channel chunks. The shared CPU reference is BatchNorm-fused and unchunked.

On November 19, 2026, batch 1 passed the Eager/Compile and CPU-reference comparisons with CPU fallback disabled and rtol=0.01, atol=0.01. The report contains 50 individual warm timings per mode.

Measured warm means:
- Eager: 11.923 ms
- Compile: 6.272 ms
- Compile speedup: 1.90x
- Maximum Eager/Compile output difference: 0.015625

Inputs were already on the NPU. Timings include output transfer to CPU. First-call measurements remain exploratory because cache state was uncontrolled.

The JSON report and execution log are stored under results/2026-11-19/ with the prefix resnet18-chunked-20261119T063813Z.

These results cover one input and one benchmark run using the working correctness tolerances.
