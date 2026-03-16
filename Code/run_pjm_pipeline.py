#!/usr/bin/env python3
"""
run_pjm_pipeline.py

Orchestrates the PJM DOM research pipeline for the current project files.

Default steps
-------------
1) merge_pjm_weather_for_model.py
2) eda_pjm_dom_weather.py
3) build_pjm_sarimax_baseline.py
4) extreme_weather_event_study.py

This wrapper is intentionally practical:
- it uses the file names already present in /mnt/data,
- it fails loudly when a required step crashes,
- it also lets you skip steps when outputs already exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path("/mnt/data")
DEFAULT_INPUT_MERGED = BASE_DIR / "pjm_dom_weather_merged_model_ready.csv"
DEFAULT_SUMMARY = BASE_DIR / "pjm_pipeline_run_summary.json"

SCRIPTS = {
    "merge": BASE_DIR / "merge_pjm_weather_for_model.py",
    "eda": BASE_DIR / "eda_pjm_dom_weather.py",
    "sarimax": BASE_DIR / "build_pjm_sarimax_baseline.py",
    "event": BASE_DIR / "extreme_weather_event_study.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PJM DOM weather-price pipeline")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Base directory containing scripts and data")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-sarimax", action="store_true", help="Skip SARIMAX baseline step")
    parser.add_argument("--skip-event", action="store_true", help="Skip extreme-weather event-study step")
    parser.add_argument("--sarimax-target", default="da_lmp", choices=["da_lmp", "rt_lmp"], help="Target for SARIMAX")
    parser.add_argument("--event-target", default="da_lmp", choices=["da_lmp", "rt_lmp"], help="Target for event study")
    parser.add_argument("--sarimax-max-rows", type=int, default=5000, help="Most recent rows used by SARIMAX script")
    parser.add_argument("--sarimax-train-frac", type=float, default=0.8, help="Training fraction used by SARIMAX")
    parser.add_argument("--sarimax-order", default="1,0,0", help="ARIMA order p,d,q for SARIMAX script")
    parser.add_argument("--sarimax-maxiter", type=int, default=40, help="Optimizer iterations for SARIMAX script")
    parser.add_argument("--event-window-days", type=int, default=14, help="Window length around Elliott event")
    parser.add_argument("--event-top-n", type=int, default=20, help="Top N summer load days for CP-like analysis")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY), help="JSON summary output path")
    return parser.parse_args()


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def run_step(name: str, cmd: List[str]) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Running step: {name}")
    print("Command:", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - start

    result: Dict[str, Any] = {
        "step": name,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (return code {proc.returncode})")

    print(f"Finished step: {name} in {elapsed:.2f}s")
    return result


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    summary_out = Path(args.summary_out)

    scripts = {k: base_dir / v.name for k, v in SCRIPTS.items()}
    for key, path in scripts.items():
        if key == "merge" and args.skip_merge:
            continue
        if key == "eda" and args.skip_eda:
            continue
        if key == "sarimax" and args.skip_sarimax:
            continue
        if key == "event" and args.skip_event:
            continue
        _check_exists(path, f"{key} script")

    steps: List[Dict[str, Any]] = []

    if not args.skip_merge:
        steps.append(
            run_step(
                "merge_pjm_weather_for_model",
                [args.python, str(scripts["merge"])],
            )
        )

    merged_file = base_dir / DEFAULT_INPUT_MERGED.name
    _check_exists(merged_file, "merged PJM-weather dataset")

    if not args.skip_eda:
        steps.append(
            run_step(
                "eda_pjm_dom_weather",
                [args.python, str(scripts["eda"]), "--input", str(merged_file)],
            )
        )

    if not args.skip_sarimax:
        steps.append(
            run_step(
                "build_pjm_sarimax_baseline",
                [
                    args.python,
                    str(scripts["sarimax"]),
                    "--input",
                    str(merged_file),
                    "--target",
                    args.sarimax_target,
                    "--max-rows",
                    str(args.sarimax_max_rows),
                    "--train-frac",
                    str(args.sarimax_train_frac),
                    "--order",
                    args.sarimax_order,
                    "--maxiter",
                    str(args.sarimax_maxiter),
                ],
            )
        )

    if not args.skip_event:
        steps.append(
            run_step(
                "extreme_weather_event_study",
                [
                    args.python,
                    str(scripts["event"]),
                    "--input",
                    str(merged_file),
                    "--target",
                    args.event_target,
                    "--window-days",
                    str(args.event_window_days),
                    "--top-n-load-days",
                    str(args.event_top_n),
                ],
            )
        )

    summary = {
        "base_dir": str(base_dir),
        "merged_file": str(merged_file),
        "steps": steps,
        "outputs": {
            "merged_csv": str(base_dir / "pjm_dom_weather_merged_model_ready.csv"),
            "merged_qa_json": str(base_dir / "pjm_dom_weather_merged_model_ready.qa.json"),
            "eda_dir": str(base_dir / "pjm_eda_outputs"),
            "sarimax_dir": str(base_dir / "pjm_sarimax_outputs"),
            "event_dir": str(base_dir / "pjm_event_study_outputs"),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"Pipeline finished. Summary written to: {summary_out}")


if __name__ == "__main__":
    main()
