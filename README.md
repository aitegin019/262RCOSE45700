# 262RCOSE45700

# PyTorch Eager vs Compile Mode on the Furiosa RNGD NPU

2026 실전SW프로젝트 — industry project with Furiosa AI.

We compare PyTorch **Eager mode** and **Compile mode** (`torch.compile` with the
Furiosa backend) for inference on the **Furiosa RNGD NPU**. For each model and
batch size we check that both modes produce matching outputs, then measure how
much faster Compile mode is.

**Target model:** ResNet-18. **Supporting models:** MLP, SmallCNN.

---

## Current Status (updated 2026-11-12)

| Area | Status | Evidence |
|---|---|---|
| Environment and BF16 operations | ✅ Working | `results/2026-10-27/` |
| SmallCNN correctness, batches 1/4/8/16 (CPU fallback off) | ✅ 60/60 comparisons pass | `results/2026-11-09/smallcnn_*` |
| SmallCNN benchmark with JSON output | ✅ Validated 11/10 | `experiments/ai-baseline/benchmark_smallcnn_json.md` |
| Original ResNet-18 on NPU Compile | ❌ Compiler error on 512-channel convolutions | `results/2026-10-27/conv512.log` |
| Modified (chunked) ResNet-18, batch 1 | ✅ 15/15 comparisons pass | `results/2026-11-09/resnet_multi_input*` |
| Modified ResNet-18, batch 4 | ❌ `insufficient sequencer entries: required 9` | `results/2026-11-09/resnet_batch4.log` |
| Final ResNet-18 benchmark | ⬜ Not started | — |

### Preliminary results (SmallCNN, October 27)

| Batch | Eager (ms) | Compile (ms) | Speedup |
|---|---|---|---|
| 1 | 0.913 | 0.424 | 2.15× |
| 4 | 1.194 | 0.643 | 1.86× |
| 8 | 1.810 | 1.100 | 1.65× |
| 16 | 2.915 | 2.009 | 1.45× |

> These timings are **preliminary**: CPU fallback was not explicitly disabled.
> Use `benchmark_smallcnn_json.py` for results that will be reported.

---

## Project Environment

| Component | Version |
|---|---|
| Hardware | Furiosa RNGD NPU (provided server) |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| furiosa-torch | 2026.4.0 |
| Precision | BF16 |

All experiments require the provided Furiosa NPU environment. Local machines
are used only for editing and Git.

---

## How to Run

Run every command from the repository root on the NPU server.

### 1. Verify the shared models

```bash
python models/verify_project_models.py --model smallcnn
python models/verify_project_models.py --model resnet18_chunked
```

### 2. Run the SmallCNN benchmark

```bash
python experiments/ai-baseline/benchmark_smallcnn_json.py --output smallcnn-results.json
```

Use a new output filename for each run. The script exits with an error if any
correctness check fails.

### 3. Use the models in your own script

```python
from models.project_models import build_model_pair, make_input

reference, model = build_model_pair("smallcnn", seed=42)   # or "resnet18_chunked"
x = make_input("smallcnn", batch=1, seed=43)
```

See [models/README.md](models/README.md) for the full Eager/Compile example.

---

## Methodology

Every experiment follows the same order: **check correctness first, then measure speed.**

- **Correctness:** the CPU reference, NPU Eager and NPU Compile outputs are
  compared pairwise with `rtol=0.01`, `atol=0.01`.
- **CPU fallback disabled:** `config.cpu_fallback(False)`, so unsupported
  operations fail visibly instead of silently running on the CPU.
- **Timing:** 10 warm-up calls, then 50 timed calls per mode. Includes the model
  call and output transfer to CPU. Excludes input transfer and first compilation.
- **Limitations:** random weights and synthetic inputs. Matching outputs show
  numerical agreement, not classification accuracy.

### ResNet-18 workaround

Convolutions with 512 output channels fail to compile. Following the mentor's
advice, `ChunkedConv2d` splits each one into four 128-channel convolutions and
concatenates the results. BatchNorm is fused into the convolutions first. Five
layers in ResNet-18 are replaced (`models/project_models.py`).

---

## Project Structure

```
262RCOSE45700/
├── models/                     # Shared, reusable models (SmallCNN, chunked ResNet-18)
│   ├── project_models.py
│   └── verify_project_models.py
├── experiments/
│   ├── ai-baseline/            # MLP/SmallCNN runs, benchmarks, correctness tests
│   └── compiler-issues/        # Conv2d and ResNet compiler error reproductions + workaround
├── results/<YYYY-MM-DD>/       # Logs and CSVs, one folder per experiment date
└── .github/workflows/          # CI
```

---

## Roadmap

- [x] Run MLP and SmallCNN in Eager and Compile modes
- [x] Reproduce the ResNet-18 compiler error and apply the chunking workaround
- [x] Validate SmallCNN and chunked ResNet-18 with CPU fallback disabled
- [x] Benchmark runner with JSON output (#4, #5)
- [ ] Resolve or document the ResNet-18 batch-4 compiler error
- [ ] Re-run SmallCNN timings with the validated runner
- [ ] ResNet-18 benchmark: latency, throughput, first-run time, memory
- [ ] Final analysis and report

---

## Team

| Name | GitHub | Role |
|---|---|---|
| TODO | @nuramirahalya | Project manager |
| TODO | @aitegin019 | Models, compiler investigation, validation |
| TODO | @aiteginkk | Benchmark runner |

---

## Team Rules

### Repository

- **Main repository: https://github.com/nuramirahalya/262RCOSE45700**
- **Integration branch:** `develop`
- **Per-student repositories:**

### Branches

| Branch | Purpose |
|---|---|
| `main` | Stable, deliverable version |
| `develop` | Integration branch |
| `feature/<issue#>-<name>` | One feature or fix per Issue |
| `model/<name>` | ML experiment branches |

### Rules

- Do not push directly to `main` or `develop`. All changes go through a Pull Request.
- Every PR needs at least 1 review and passing CI checks before merging.
- One Issue per task, one purpose per PR.
- Commit messages describe what changed (no `update` or `final`).
- Every member commits under their own GitHub account.
- AI-generated code must be reviewed and understood by a team member before committing.
- Never commit `.venv/`, `__pycache__/`, `.env`, API keys or large raw result files.
- Each experiment PR includes the script, its log/CSV in `results/<date>/`, and a note on what was tested and its limits.
