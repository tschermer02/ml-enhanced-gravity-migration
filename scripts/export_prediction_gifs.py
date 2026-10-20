"""Export GIF-only 3D Plotly visualizations from saved U-Net predictions.

Use the existing NPZ predictions; do not rerun model inference or create HTML.
Run from the repository root:
    python scripts/export_prediction_gifs.py --experiment all --frames 24

Requires: plotly, kaleido, pillow, and compatible Chrome/Chromium.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualization.plotly_3d import plot_sample_3d

EXPERIMENTS = ("E01", "E02_balanced_huber", "E03", "E04")
EXPECTED_SHAPE = (24, 64, 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="all")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--predictions-root", type=Path,
                        default=ROOT / "outputs" / "ml" / "predict")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Animation duration, in seconds.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum samples per experiment; default: all.")
    parser.add_argument("--reconstruction-threshold", type=float, default=0.25,
                        help="Fraction of predicted peak used to display isosurfaces.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate existing GIFs. Default is to skip them.")
    args = parser.parse_args()
    if args.frames < 2 or args.duration <= 0:
        parser.error("--frames >= 2 and --duration > 0 are required")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if not 0 < args.reconstruction_threshold < 1:
        parser.error("--reconstruction-threshold must be between 0 and 1")

    root = args.predictions_root
    if not root.is_absolute():
        root = ROOT / root
    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)

    for exp in experiments:
        prediction_dir = root / exp / "predictions" / args.split
        files = sorted(prediction_dir.glob("*.npz"))
        if args.limit is not None:
            files = files[:args.limit]
        if not files:
            print(f"Skipping {exp}: no saved predictions in {prediction_dir}", flush=True)
            continue
        gif_dir = root / exp / "figures" / args.split
        gif_dir.mkdir(parents=True, exist_ok=True)
        for i, archive_path in enumerate(files, 1):
            gif_path = gif_dir / f"{archive_path.stem}_3d.gif"
            if gif_path.exists() and not args.overwrite:
                print(f"Skipping existing: {gif_path}", flush=True)
                continue
            with np.load(archive_path) as archive:
                true_density = np.asarray(archive["density"], dtype=np.float32)
                prediction = np.asarray(archive["prediction"], dtype=np.float32)
            if true_density.shape != EXPECTED_SHAPE or prediction.shape != EXPECTED_SHAPE:
                raise ValueError(f"Unexpected array shape in {archive_path}")
            if not np.isfinite(true_density).all() or not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite array in {archive_path}")

            print(f"{exp} {i}/{len(files)}: generating {gif_path.name}", flush=True)
            # Gravity is currently unused by plot_sample_3d. The dummy argument
            # keeps compatibility with its existing signature; no forward model runs.
            unused_gravity = np.empty((0,), dtype=np.float32)
            plot_sample_3d(
                true_density,
                unused_gravity,
                gif_path,  # .gif triggers GIF-only export: NO .html file is written
                reconstruction=prediction,
                reconstruction_threshold=args.reconstruction_threshold,
                make_gif=True,
                gif_frames=args.frames,
                gif_duration_seconds=args.duration,
            )
            print(f"Saved: {gif_path}", flush=True)


if __name__ == "__main__":
    main()
