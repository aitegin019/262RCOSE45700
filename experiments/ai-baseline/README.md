# Initial Furiosa NPU experiments

Author: Aitegin Altynbekova (2022320316)

Experiments were performed on October 27, 2026.
Scripts and recorded logs were imported into Git afterwards.

## Environment

- Furiosa RNGD NPU
- Python 3.12.3
- PyTorch 2.10.0+cpu
- torchvision 0.25.0+cpu
- furiosa-torch 2026.4.0
- BF16 precision

Run these scripts in the provided Furiosa environment.
The Mac is used for Git and editing; these scripts require NPU access.

## Run from the repository root

Basic inference:

    python experiments/ai-baseline/first_model.py

MLP Eager/Compile output comparison:

    python experiments/ai-baseline/compare_modes.py

SmallCNN output comparison:

    python experiments/ai-baseline/test_small_cnn.py

SmallCNN batch-1 latency benchmark:

    python experiments/ai-baseline/benchmark_cnn.py

SmallCNN batch-size benchmark with output checks:

    python experiments/ai-baseline/benchmark_batch_verified.py

## Recorded results

Logs are stored in results/2026-10-27/.

The verified batch experiment reported:
- Batch 1: 2.15x speedup
- Batch 4: 1.86x speedup
- Batch 8: 1.65x speedup
- Batch 16: 1.45x speedup

These are preliminary measurements for this model and configuration.

## Measurement method and limitations

- 10 warm-up calls and 50 timed calls per mode.
- Inputs are already on the NPU before timing starts.
- Timing includes the model call and output transfer to CPU.
- Input transfer and first-run compilation costs are excluded.
- CPU fallback behavior has not yet been explicitly verified.
- Output comparisons use rtol=1e-2 and atol=1e-2.
- The scripts currently print correctness failures without stopping.
- Models use random weights and synthetic inputs.
- Output agreement does not establish classification accuracy.

## Compiler investigation

Related scripts are in experiments/compiler-issues/.

Recorded standalone convolution tests succeeded at 64, 128, and
256 input/output channels. The 512-channel test failed during graph
compilation with TransposedLower/LowLevelTailCrop errors.

The mentor suggested splitting output channels into chunks of 128
and concatenating the outputs. This workaround has not yet been
validated in these recorded experiments.

## AI assistance

ChatGPT assisted with code drafting, debugging guidance, and Git
workflow instructions. I executed the experiments on the provided
NPU, collected results, and investigated the compilation failures.
