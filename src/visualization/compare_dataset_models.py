"""Plot a synthetic density model beside its migrated volume."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualization.plotting import plot_volume_comparison  # noqa: E402

DATASET_ROOT = SRC / "data" / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--sample-id", default="sample_000000")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="PNG output path. Defaults to results/figures/dataset_comparisons/<split>_<sample>.png.",
    )
    args = parser.parse_args()

    synthetic_path = DATASET_ROOT / "synthetic_models" / args.split / f"{args.sample_id}.npz"
    migration_path = DATASET_ROOT / "migration_models" / args.split / f"{args.sample_id}.npz"
    if not synthetic_path.is_file():
        raise FileNotFoundError(f"Synthetic model not found: {synthetic_path}")
    if not migration_path.is_file():
        raise FileNotFoundError(f"Migration model not found: {migration_path}")

    with np.load(synthetic_path) as archive:
        density = np.asarray(archive["density"], dtype=np.float64)
    with np.load(migration_path) as archive:
        migration = np.asarray(archive["migration"], dtype=np.float64)

    if density.shape != migration.shape or density.ndim != 3:
        raise ValueError(
            f"Expected matching 3D volumes; got density {density.shape} and migration {migration.shape}"
        )
    if not np.isfinite(density).all() or not np.isfinite(migration).all():
        raise ValueError("Density and migration volumes must contain only finite values")

    output = args.output
    if output is None:
        output = Path("results/figures/dataset_comparisons") / f"{args.split}_{args.sample_id}.png"
    elif not output.is_absolute():
        output = ROOT / output

    plot_volume_comparison(
        density,
        migration,
        output,
        estimate_label="Migration model",
        estimate_units="Migration amplitude (arbitrary units)",
        title=f"{args.split.title()} comparison: {args.sample_id}",
    )
    print(f"Saved comparison plot: {output}")


if __name__ == "__main__":
    main()
