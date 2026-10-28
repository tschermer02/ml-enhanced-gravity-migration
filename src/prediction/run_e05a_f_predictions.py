"""Run the pretrained E05 network on 100 randomized samples per E05A--E05F case.

Generate datasets first:
  python scripts/generate_e05a_f_cases.py --samples 100

Run evaluation (default: all available predictions, 3 figures per case):
  python scripts/run_e05a_f_predictions.py
  python scripts/run_e05a_f_predictions.py --cases E05A E05F --visualize-count 5
  python scripts/run_e05a_f_predictions.py --visualize-count 0
  python scripts/run_e05a_f_predictions.py --limit 10 --visualize-count 2

All cases use the SAME E05 checkpoint. Metrics are per-case, not pooled.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("E05A", "E05B", "E05C", "E05D", "E05E", "E05F")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--evaluation-root", type=Path, default=Path("outputs/ml/evaluation"))
    parser.add_argument("--prediction-root", type=Path, default=Path("outputs/ml/predict"))
    parser.add_argument("--model", type=Path,
                        default=Path("outputs/ml/training/E05/checkpoints/best_unet3d.keras"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process first N samples per case; default: all (100).")
    parser.add_argument("--visualize-count", type=int, default=3,
                        help="Number of density/gravity PNG pairs per case (0 = none).")
    parser.add_argument("--plotly", action="store_true",
                        help="Also make Plotly GIFs for visualized samples.")
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing to existing per-case prediction directories.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.visualize_count < 0:
        parser.error("--visualize-count must be >= 0")
    if args.gif_frames < 2:
        parser.error("--gif-frames must be >= 2")

    evaluation_root = resolve(args.evaluation_root)
    prediction_root = resolve(args.prediction_root)
    model = resolve(args.model)
    if not model.is_file():
        raise FileNotFoundError(f"Missing trained E05 checkpoint: {model}")

    for case in dict.fromkeys(args.cases):
        data = evaluation_root / case / "data"
        gravity = evaluation_root / case / "forward_models"
        output = prediction_root / case
        files = sorted((data / "test").glob("*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No preprocessed test data found under {data / 'test'}. "
                "Run scripts/generate_e05a_f_cases.py first."
            )
        count = min(len(files), args.limit) if args.limit else len(files)
        command = [
            sys.executable, str(ROOT / "scripts" / "predict_unet.py"),
            "--experiment", "E05", "--split", "test",
            "--model", str(model),
            "--data-root", str(data),
            "--gravity-root", str(gravity),
            "--output-root", str(output),
            "--visualize-count", str(args.visualize_count),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.plotly:
            command.extend(["--plotly", "--gif-frames", str(args.gif_frames)])
        if args.overwrite:
            command.append("--overwrite")
        print(f"\n{'='*65}\n{case}: evaluating {count} samples\n{'='*65}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        print(f"{case}: {output / 'predictions' / 'test_summary.json'}", flush=True)

    print("\nFinished selected E05 challenge evaluations.", flush=True)


if __name__ == "__main__":
    main()
