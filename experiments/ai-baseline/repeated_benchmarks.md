# Repeated inference benchmarks

Run from the repository root in the provided Furiosa NPU environment:

    python experiments/ai-baseline/repeat_benchmarks.py --output-dir repeated-results

The default runs SmallCNN and chunked ResNet-18 five times each in separate, sequential processes. SmallCNN uses batches 1, 4, 8, and 16. ResNet uses batch 1.

Optional arguments:

    --repetitions 5
    --models smallcnn
    --models resnet18_chunked

Use a new output directory for each experiment. Each process saves a JSON report and execution log. manifest.json records the model, repetition, exit code, and filenames. A failed process is recorded, the remaining processes are attempted, and the tool exits unsuccessfully if any process failed.

Generate the summary on the server or on a computer with Python:

    python experiments/ai-baseline/summarize_benchmarks.py repeated-results

The summary checks expected repetitions, batch sizes, CPU-fallback settings, correctness results, and finite positive timing samples.

Each model/batch/mode has 250 warm samples across five runs. Mean, p50, and p95 are calculated from pooled samples. The p95 uses inclusive linear interpolation. Throughput is batch size times 1000 divided by mean latency in milliseconds. Run-mean CV is the sample standard deviation of the five run means divided by their mean, expressed as a percentage.

Inputs are already on the NPU. Timings include model execution and output transfer to CPU. CPU fallback is disabled. The shared models use random BF16 weights and synthetic inputs.

Fresh processes do not control persistent compiler cache state. First-call measurements remain exploratory.

## November 19 results

Reports, logs, manifest, and summary are under results/2026-11-19/repeated-20261119T065641Z/.

All ten processes completed successfully and their correctness checks passed. The reports contain 2500 individual warm timings. Run-mean CV ranged from 0.05% to 1.22%.

| Model | Batch | Eager mean (ms) | Compile mean (ms) |
|---|---:|---:|---:|
| SmallCNN | 1 | 0.878 | 0.426 |
| SmallCNN | 4 | 1.181 | 0.646 |
| SmallCNN | 8 | 1.812 | 1.100 |
| SmallCNN | 16 | 2.906 | 1.979 |
| ResNet-18 chunked | 1 | 11.955 | 6.265 |

Full precision values, p50, p95, throughput, and run means are in summary.json.
