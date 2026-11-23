"""Run benchmark repetitions and preserve reports and execution logs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=5)
parser.add_argument(
    "--models",
    nargs="+",
    choices=["smallcnn", "resnet18_chunked"],
    default=["smallcnn", "resnet18_chunked"],
)
args = parser.parse_args()

if args.repetitions < 2:
    parser.error("--repetitions must be at least 2")

models = list(dict.fromkeys(args.models))
args.output_dir.mkdir(parents=True, exist_ok=False)
runner = Path(__file__).with_name("benchmark_smallcnn_json.py").resolve()
runs = []

for model in models:
    for repetition in range(1, args.repetitions + 1):
        name = f"{model}-run-{repetition:02d}"
        report_path = args.output_dir / f"{name}.json"
        log_path = args.output_dir / f"{name}.log"
        print(f"Running {name}...", flush=True)

        command = [
            sys.executable,
            "-u",
            str(runner),
            "--model",
            model,
            "--output",
            str(report_path.resolve()),
        ]

        with log_path.open("x") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        runs.append({
            "model": model,
            "repetition": repetition,
            "exit_code": completed.returncode,
            "report": report_path.name if report_path.exists() else None,
            "log": log_path.name,
        })

        manifest = {
            "repetitions": args.repetitions,
            "models": models,
            "protocol": "Sequential fresh processes; persistent cache uncontrolled",
            "runs": runs,
        }
        manifest_path = args.output_dir / "manifest.json"
        temporary_path = args.output_dir / "manifest.json.tmp"
        temporary_path.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_path.replace(manifest_path)

        print(f"Finished {name}: exit code {completed.returncode}", flush=True)

if any(run["exit_code"] != 0 for run in runs):
    raise SystemExit("Some runs failed. Inspect manifest.json and the logs.")

print(f"Completed {len(runs)} runs. Results saved in {args.output_dir}")
