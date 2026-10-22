"""Evaluate one trained E05 checkpoint separately on challenge cases E05A--E05F.

This does NOT retrain E05. It invokes the repository's existing
scripts/predict_unet.py with per-case data/output paths.

Place at scripts/run_e05a_f_predictions.py; from repository root:
    python scripts/run_e05a_f_predictions.py
    python scripts/run_e05a_f_predictions.py --cases E05A E05C E05F
    python scripts/run_e05a_f_predictions.py --cases E05A --visualize-count 0

Generate inputs first with scripts/generate_e05a_f_cases.py.
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
    parser.add_argument("--evaluation-root", type=Path,
                        default=Path("outputs/ml/evaluation"))
    parser.add_argument("--prediction-root", type=Path,
                        default=Path("outputs/ml/predict"))
    parser.add_argument("--model", type=Path,
                        default=Path("outputs/ml/training/E05/checkpoints/best_unet3d.keras"))
    parser.add_argument("--visualize-count", type=int, default=1,
                        help="Per case: 0 disables density/gravity figures; 1 enables both.")
    parser.add_argument("--plotly", action="store_true",
                        help="Also render GIFs with Plotly/Kaleido (requires Chrome).")
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit overwriting case prediction output directories.")
    args = parser.parse_args()
    if args.visualize_count not in (0, 1):
        parser.error("--visualize-count must be 0 or 1 (each case has one sample)")
    if args.gif_frames < 2:
        parser.error("--gif-frames must be at least 2")

    evaluation_root = resolve(args.evaluation_root)
    prediction_root = resolve(args.prediction_root)
    model_path = resolve(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained E05 checkpoint not found: {model_path}")

    for case in dict.fromkeys(args.cases):
        data = evaluation_root / case / "data"
        gravity = evaluation_root / case / "forward_models"
        pred = prediction_root / case
        check = data / "test/sample_000000.npz"
        if not check.is_file():
            raise FileNotFoundError(
                f"Missing input {check}\n"
                "First run: python scripts/generate_e05a_f_cases.py"
            )
        command = [
            sys.executable, str(ROOT / "scripts/predict_unet.py"),
            "--experiment", "E05", "--split", "test",
            "--model", str(model_path),
            "--data-root", str(data),
            "--gravity-root", str(gravity),
            "--output-root", str(pred),
            "--visualize-count", str(args.visualize_count),
        ]
        if args.plotly:
            command += ["--plotly", "--gif-frames", str(args.gif_frames)]
        if args.overwrite:
            command.append("--overwrite")
        print(f"\n{'=' * 60}\n{case}: running trained E05 on "
              f"{(evaluation_root / case).name}\n{'=' * 60}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        print(f"{case}: metrics saved to {pred / 'predictions/test_summary.json'}", flush=True)

    print("\nFinished all selected E05 geological challenge cases.", flush=True)


if __name__ == "__main__":
    main()
