"""Export rotating Plotly GIFs from previously saved U-Net density predictions.

Supports E01-E06 and signed-density E06 categories. Does not run inference.

Examples:
  python scripts/export_prediction_gifs.py --experiment E06_signed/E06D --samples 1 4 9 --overwrite
  python scripts/export_prediction_gifs.py --experiment E06/E06B --samples 13 --frames 24
  python scripts/export_prediction_gifs.py --experiment all --limit 1
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

EXPERIMENTS = (
    "E01", "E02_balanced_huber", "E03", "E04", "E04B", "E05",
    "E06/E06A", "E06/E06B", "E06/E06C", "E06/E06D", "E06/E06E", "E06/E06F",
    "E06_signed/E06A", "E06_signed/E06B", "E06_signed/E06C",
    "E06_signed/E06D", "E06_signed/E06E", "E06_signed/E06F",
)
EXPECTED_SHAPE = (24, 64, 64)


def normalize_sample_name(sample: str) -> str:
    """Accept 1, 000001, sample_000001, or sample_000001.npz."""
    stem = sample.strip().removesuffix(".npz")
    if stem.startswith("sample_"):
        stem = stem[len("sample_"):]
    if stem.isdigit():
        return f"sample_{int(stem):06d}.npz"
    return f"{stem}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="all")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--predictions-root", type=Path,
                        default=ROOT / "outputs" / "ml" / "predict")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reconstruction-threshold", type=float, default=0.25,
                        help="Fraction of peak predicted absolute contrast for 3D isosurfaces")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Sample IDs: e.g. 1 4 9, or sample_000001")
    args = parser.parse_args()

    if args.frames < 2:
        parser.error("--frames must be >= 2")
    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if not 0 < args.reconstruction_threshold < 1:
        parser.error("--reconstruction-threshold must be between 0 and 1")

    root = args.predictions_root
    if not root.is_absolute():
        root = ROOT / root
    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    requested = ({normalize_sample_name(s) for s in args.samples}
                 if args.samples else None)
    exported = 0

    for experiment in experiments:
        prediction_dir = root / experiment / "predictions" / args.split
        if not prediction_dir.is_dir():
            print(f"Skipping {experiment}: not found: {prediction_dir}", flush=True)
            continue

        files = sorted(prediction_dir.glob("*.npz"))
        if requested is not None:
            available = {p.name for p in files}
            missing = requested - available
            if missing:
                print(f"{experiment}: requested samples not found: {sorted(missing)}", flush=True)
            files = [p for p in files if p.name in requested]
        if args.limit is not None:
            files = files[:args.limit]
        if not files:
            print(f"Skipping {experiment}: no matching predictions", flush=True)
            continue

        gif_dir = root / experiment / "figures" / args.split
        gif_dir.mkdir(parents=True, exist_ok=True)
        for index, archive_path in enumerate(files, 1):
            gif_path = gif_dir / f"{archive_path.stem}_3d.gif"
            if gif_path.is_file() and not args.overwrite:
                print(f"Skipping existing: {gif_path}", flush=True)
                continue
            with np.load(archive_path) as archive:
                truth = np.asarray(archive["density"], dtype=np.float32)
                prediction = np.asarray(archive["prediction"], dtype=np.float32)
            for label, array in (("density", truth), ("prediction", prediction)):
                if array.shape != EXPECTED_SHAPE or not np.isfinite(array).all():
                    raise ValueError(f"{archive_path}: invalid {label} ({array.shape})")

            print(f"{experiment} {index}/{len(files)}: {gif_path.name}", flush=True)
            # The existing plot function accepts gravity but does not use it.
            plot_sample_3d(
                truth, np.empty((0,), dtype=np.float32), gif_path,
                reconstruction=prediction,
                reconstruction_threshold=args.reconstruction_threshold,
                make_gif=True, gif_frames=args.frames,
                gif_duration_seconds=args.duration,
            )
            exported += 1
            print(f"Saved: {gif_path}", flush=True)

    print(f"Completed: {exported} GIF(s) exported.", flush=True)


if __name__ == "__main__":
    main()
