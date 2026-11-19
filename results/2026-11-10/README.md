# SmallCNN benchmark — November 10, 2026

Report: [smallcnn-20261110T080053Z.json](smallcnn-20261110T080053Z.json)

The run used the shared SmallCNN model with random BF16 weights, model seed 42, and input seed 43 on Furiosa RNGD. CPU fallback was disabled.

All four batches passed Eager/Compile and CPU-reference output comparisons at rtol=0.01 and atol=0.01. The JSON contains 400 individual warm timings.

| Batch | Eager mean (ms) | Compile mean (ms) | Speedup |
|---:|---:|---:|---:|
| 1 | 0.940 | 0.431 | 2.18x |
| 4 | 1.172 | 0.645 | 1.82x |
| 8 | 1.805 | 1.096 | 1.65x |
| 16 | 2.905 | 1.993 | 1.46x |

Each mode used 10 warm-up calls followed by 50 timed calls per batch. Input was already on the NPU. Timings include model execution and output transfer to CPU.

First-call values were measured per batch shape in one process, with Eager first and uncontrolled cache state. They remain exploratory.

These results cover one input per batch and one benchmark run using working BF16 tolerances.

Environment: Python 3.12.3, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu, furiosa-torch 2026.4.0.
