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
