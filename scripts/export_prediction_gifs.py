"""Export GIF-only 3D Plotly visualizations from saved U-Net predictions.

Use the existing NPZ predictions; do not rerun model inference or create HTML.

Run from the repository root:
    python scripts/export_prediction_gifs.py --experiment all --frames 24

Examples:
    # Export all saved E04 test GIFs
    python scripts/export_prediction_gifs.py --experiment E04

    # Export only specific E04 samples
    python scripts/export_prediction_gifs.py --experiment E04 --samples sample_000006 sample_000051

    # Export only the first 3 E04 samples
    python scripts/export_prediction_gifs.py --experiment E04 --limit 3

Requires: plotly, kaleido, pillow, and compatible Chrome/Chromium.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualization.plotly_3d import plot_sample_3d

# ============================================================
# CONSTANTS
# ============================================================

EXPERIMENTS = (
    "E01",
    "E02_balanced_huber",
    "E03",
    "E04",
    "E04B",
    "E05",
    "E06/E06A",
    "E06/E06B",
    "E06/E06C",
    "E06/E06D",
    "E06/E06E",
    "E06/E06F",
)
EXPECTED_SHAPE = (24, 64, 64)


# ============================================================
# HELPERS
# ============================================================

def normalize_sample_name(sample: str) -> str:
    """
    Convert a user-provided sample identifier into a canonical NPZ filename.

    Accepted forms:
        6
        000006
        sample_000006
        sample_000006.npz
    """
    sample = sample.strip()

    if sample.endswith(".npz"):
        sample = sample[:-4]

    if sample.startswith("sample_"):
        suffix = sample[len("sample_"):]
        if suffix.isdigit():
            return f"sample_{int(suffix):06d}.npz"
        return f"{sample}.npz"

    if sample.isdigit():
        return f"sample_{int(sample):06d}.npz"

    return f"{sample}.npz"


def select_prediction_files(prediction_dir: Path, requested_samples: list[str] | None) -> list[Path]:
    """
    Gather saved prediction files and optionally filter to specific samples.
    """
    files = sorted(prediction_dir.glob("*.npz"))

    if requested_samples is None:
        return files

    requested = {normalize_sample_name(sample) for sample in requested_samples}
    return [path for path in files if path.name in requested]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--experiment",
        choices=(*EXPERIMENTS, "all"),
        default="all",
        help="Which experiment to export GIFs for.",
    )

    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Dataset split to read saved predictions from.",
    )

    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=ROOT / "outputs" / "ml" / "predict",
        help="Root directory containing saved prediction folders.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=24,
        help="Number of frames in the rotating GIF.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Animation duration in seconds.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples per experiment after filtering.",
    )

    parser.add_argument(
        "--reconstruction-threshold",
        type=float,
        default=0.25,
        help="Fraction of predicted peak used to display reconstruction isosurfaces.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing GIFs instead of skipping them.",
    )

    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help=(
            "Specific sample IDs or filenames to export, e.g. "
            "sample_000006 sample_000051 or 6 51."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.frames < 2:
        parser.error("--frames must be at least 2")

    if args.duration <= 0:
        parser.error("--duration must be > 0")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    if not 0 < args.reconstruction_threshold < 1:
        parser.error("--reconstruction-threshold must be between 0 and 1")

    # --------------------------------------------------------
    # Resolve root directory
    # --------------------------------------------------------

    predictions_root = args.predictions_root
    if not predictions_root.is_absolute():
        predictions_root = ROOT / predictions_root

    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)

    # --------------------------------------------------------
    # Export GIFs
    # --------------------------------------------------------

    for exp in experiments:
        prediction_dir = predictions_root / exp / "predictions" / args.split

        if not prediction_dir.is_dir():
            print(f"Skipping {exp}: prediction directory not found: {prediction_dir}", flush=True)
            continue

        files = select_prediction_files(prediction_dir, args.samples)

        if args.limit is not None:
            files = files[:args.limit]

        if not files:
            if args.samples:
                print(
                    f"Skipping {exp}: no matching saved predictions found in {prediction_dir} "
                    f"for requested samples {args.samples}",
                    flush=True,
                )
            else:
                print(f"Skipping {exp}: no saved predictions in {prediction_dir}", flush=True)
            continue

        gif_dir = predictions_root / exp / "figures" / args.split
        gif_dir.mkdir(parents=True, exist_ok=True)

        for i, archive_path in enumerate(files, 1):
            gif_path = gif_dir / f"{archive_path.stem}_3d.gif"

            if gif_path.exists() and not args.overwrite:
                print(f"Skipping existing: {gif_path}", flush=True)
                continue

            # ------------------------------------------------
            # Load saved truth and prediction arrays
            # ------------------------------------------------

            with np.load(archive_path) as archive:
                true_density = np.asarray(archive["density"], dtype=np.float32)
                prediction = np.asarray(archive["prediction"], dtype=np.float32)

            if true_density.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Unexpected true-density shape in {archive_path}: "
                    f"{true_density.shape}, expected {EXPECTED_SHAPE}"
                )

            if prediction.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Unexpected prediction shape in {archive_path}: "
                    f"{prediction.shape}, expected {EXPECTED_SHAPE}"
                )

            if not np.isfinite(true_density).all():
                raise ValueError(f"Non-finite true density values found in {archive_path}")

            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite prediction values found in {archive_path}")

            print(
                f"{exp} {i}/{len(files)}: generating {gif_path.name}",
                flush=True,
            )

            # Gravity is currently unused by plot_sample_3d.
            # This dummy array keeps compatibility with the existing function signature.
            unused_gravity = np.empty((0,), dtype=np.float32)

            # Passing a .gif output path triggers GIF export only.
            # No HTML file is written.
            plot_sample_3d(
                true_density,
                unused_gravity,
                gif_path,
                reconstruction=prediction,
                reconstruction_threshold=args.reconstruction_threshold,
                make_gif=True,
                gif_frames=args.frames,
                gif_duration_seconds=args.duration,
            )

            print(f"Saved: {gif_path}", flush=True)


if __name__ == "__main__":
    main()