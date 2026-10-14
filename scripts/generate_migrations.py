"""Precompute raw and normalized migration inputs for the existing data splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physics.gravity_migration import calculate_migration  # noqa: E402

DATASET_ROOT = SRC / "data" / "dataset"
DEFAULT_OUTPUT = ROOT / "outputs" / "ml" / "datasets"
VOLUME_SHAPE = (24, 64, 64)
GRAVITY_SHAPE = (81, 81)


def normalize_migration(migration: np.ndarray,
                        epsilon: float = 1e-8) -> tuple[np.ndarray, float]:
    """Scale one migration volume by its max absolute value; return raw scale."""
    migration = np.asarray(migration, dtype=np.float64)
    if not np.isfinite(migration).all():
        raise ValueError("Migration contains NaN or infinity")
    scale = float(np.max(np.abs(migration)))
    normalized = migration / max(scale, epsilon)
    return normalized.astype(np.float32), scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N samples in each selected split.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
    summary: dict[str, object] = {"splits": {}, "restrict": False, "normalization": "max_abs_per_sample"}

    for split in splits:
        synthetic_dir = dataset_root / "synthetic_models" / split
        forward_dir = dataset_root / "forward_models" / split
        destination_dir = output_root / split
        synthetic_files = sorted(synthetic_dir.glob("*.npz"))
        if args.limit is not None:
            synthetic_files = synthetic_files[:args.limit]
        if not synthetic_files:
            raise FileNotFoundError(f"No synthetic models found in {synthetic_dir}")
        destination_dir.mkdir(parents=True, exist_ok=True)

        for index, synthetic_path in enumerate(synthetic_files, 1):
            forward_path = forward_dir / synthetic_path.name
            if not forward_path.is_file():
                raise FileNotFoundError(f"Forward model missing: {forward_path}")
            with np.load(synthetic_path) as synthetic:
                density = np.asarray(synthetic["density"], dtype=np.float64)
            with np.load(forward_path) as forward:
                gravity = np.asarray(forward["gravity"], dtype=np.float64)
            if density.shape != VOLUME_SHAPE or gravity.shape != GRAVITY_SHAPE:
                raise ValueError(
                    f"{split}/{synthetic_path.name}: got density {density.shape} "
                    f"and gravity {gravity.shape}"
                )
            if not np.isfinite(density).all() or not np.isfinite(gravity).all():
                raise ValueError(f"{split}/{synthetic_path.name}: nonfinite input")

            migration = calculate_migration(gravity, restrict=False)
            migration_input, migration_scale = normalize_migration(migration)
            np.savez_compressed(
                destination_dir / synthetic_path.name,
                migration=migration.astype(np.float32),
                migration_input=migration_input,
                migration_scale=np.float64(migration_scale),
                density=density.astype(np.float32),
            )
            print(f"Preprocessed {split} {index}/{len(synthetic_files)}: "
                  f"{synthetic_path.stem}", flush=True)

        summary["splits"][split] = len(synthetic_files)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
