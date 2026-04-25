from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def run_step(script_name: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if extra_args:
        command.extend(extra_args)
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_DIR.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full PJM weather 24h forecasting pipeline."
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip expanding-window cross-validation during model evaluation.",
    )
    parser.add_argument(
        "--origin-step-hours",
        type=int,
        default=24,
        help="Spacing between forecast origins for holdout evaluation.",
    )
    parser.add_argument(
        "--run-sarimax",
        action="store_true",
        help="Run the optional SARIMAX baseline. This can be slow.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip report figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_step("build_model_dataset.py")

    train_args = ["--origin-step-hours", str(args.origin_step_hours)]
    if args.skip_cv:
        train_args.append("--skip-cv")
    if args.run_sarimax:
        train_args.extend(
            ["--run-sarimax", "--sarimax-origin-step-hours", "168", "--max-sarimax-origins", "60"]
        )
    run_step("train_24h_models.py", train_args)

    run_step("analyze_winter_summer_patterns.py")

    if not args.skip_figures:
        run_step("generate_report_figures.py")

    print("\nPipeline finished. See the processed/ and reports/ folders.")


if __name__ == "__main__":
    main()
