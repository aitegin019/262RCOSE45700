"""Validate repeated benchmark reports and summarize warm timings."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()

manifest = json.loads((args.directory / "manifest.json").read_text())
repetitions = manifest["repetitions"]
expected = {
    (model, repetition)
    for model in manifest["models"]
    for repetition in range(1, repetitions + 1)
}
seen = set()
groups = defaultdict(list)

for run in manifest["runs"]:
    key = (run["model"], run["repetition"])
    if key not in expected or key in seen:
        raise SystemExit(f"Unexpected or duplicate run: {key}")
    seen.add(key)
    if run["exit_code"] != 0 or not run["report"]:
        raise SystemExit(f"Failed run: {key}; inspect {run['log']}")

    report = json.loads((args.directory / run["report"]).read_text())
    expected_model = "SmallCNN" if run["model"] == "smallcnn" else run["model"]
    expected_batches = [1, 4, 8, 16] if run["model"] == "smallcnn" else [1]

    if report["model"] != expected_model or report["cpu_fallback"] is not False:
        raise SystemExit(f"Unexpected model or fallback setting: {key}")
    if [row["batch_size"] for row in report["results"]] != expected_batches:
        raise SystemExit(f"Unexpected batches: {key}")

    for row in report["results"]:
        if row["status"] != "passed" or not all(
            row[field] for field in (
                "outputs_close", "eager_matches_cpu", "compile_matches_cpu"
            )
        ):
            raise SystemExit(f"Correctness check failed: {key}")

        for mode in ("eager", "compile"):
            samples = row[f"{mode}_warm_samples_ms"]
            if len(samples) != report["timed_iterations"] or not samples:
                raise SystemExit(f"Missing timings: {key}, {mode}")
            if any(not math.isfinite(value) or value <= 0 for value in samples):
                raise SystemExit(f"Invalid timings: {key}, {mode}")
            groups[(run["model"], row["batch_size"], mode)].append(samples)

if seen != expected:
    raise SystemExit("Some expected repetitions are missing.")

summary = []
for (model, batch, mode), runs in sorted(groups.items()):
    samples = [value for run in runs for value in run]
    run_means = [statistics.mean(run) for run in runs]
    mean_ms = statistics.mean(samples)
    row = {
        "model": model,
        "batch_size": batch,
        "mode": mode,
        "runs": len(runs),
        "samples": len(samples),
        "warm_mean_ms": mean_ms,
        "warm_p50_ms": statistics.median(samples),
        "warm_p95_ms": statistics.quantiles(samples, n=100, method="inclusive")[94],
        "throughput_images_per_second": batch * 1000 / mean_ms,
        "run_means_ms": run_means,
        "run_mean_cv_percent": statistics.stdev(run_means)
        / statistics.mean(run_means) * 100,
    }
    summary.append(row)
    print(
        f"{model} batch {batch} {mode}: "
        f"mean={mean_ms:.3f} ms, p50={row['warm_p50_ms']:.3f} ms, "
        f"p95={row['warm_p95_ms']:.3f} ms, "
        f"throughput={row['throughput_images_per_second']:.1f} images/s, "
        f"run-mean CV={row['run_mean_cv_percent']:.2f}%"
    )

output = args.directory / "summary.json"
temporary = args.directory / "summary.json.tmp"
temporary.write_text(json.dumps({
    "source_manifest": "manifest.json",
    "percentile_method": "Pooled warm samples; linear interpolation, inclusive",
    "variation_method": "Sample standard deviation of run means / their mean",
    "throughput_scope": "Sequential batches; model execution and output copy to CPU",
    "results": summary,
}, indent=2, allow_nan=False) + "\n")
temporary.replace(output)
print(f"Saved {output}")
