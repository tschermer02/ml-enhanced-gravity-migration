"""Generate separate forward-gravity and migration datasets from synthetic models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physics.gravity_forward import calculate_gravity  # noqa: E402
from physics.gravity_migration import calculate_migration  # noqa: E402


def process_sample(source_text: str, forward_text: str,
                   migration_text: str) -> dict[str, object]:
    """Run forward modeling and migration for one synthetic density model."""
    source = Path(source_text)
    forward_path = Path(forward_text)
    migration_path = Path(migration_text)
    started = time.perf_counter()

    with np.load(source) as archive:
        density = np.asarray(archive["density"], dtype=np.float64)
    if density.shape != (24, 64, 64):
        raise ValueError(f"{source.name}: density shape is {density.shape}, expected (24, 64, 64)")
    if not np.isfinite(density).all():
        raise ValueError(f"{source.name}: density contains NaN or infinity")

    gravity = calculate_gravity(density)
    migration = calculate_migration(gravity)
    forward_path.parent.mkdir(parents=True, exist_ok=True)
    migration_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(forward_path, gravity=gravity.astype(np.float32))
    np.savez_compressed(migration_path, migration=migration.astype(np.float32))

    return {
        "split": source.parent.name,
        "sample_id": source.stem,
        "synthetic_model": str(source.relative_to(ROOT)),
        "forward_model": str(forward_path.relative_to(ROOT)),
        "migration_model": str(migration_path.relative_to(ROOT)),
        "migration_min": float(np.min(migration)),
        "migration_max": float(np.max(migration)),
        "runtime_seconds": time.perf_counter() - started,
        "success": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("src/data/dataset/synthetic_models"),
        help="Synthetic-model root, split directory, or directory containing NPZ models.",
    )
    parser.add_argument("--split", choices=("all", "train", "validation", "test"),
                        default="all")
    parser.add_argument("--output-root", type=Path, default=Path("src/data/dataset"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit each selected split to its first N models.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if any(dataset.glob("*.npz")):
        split_directories = {dataset.name: dataset}
    else:
        splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
        split_directories = {split: dataset / split for split in splits}

    samples: list[tuple[str, Path]] = []
    for split, split_dir in split_directories.items():
        split_files = sorted(split_dir.glob("*.npz"))
        if args.limit is not None:
            split_files = split_files[:args.limit]
        samples.extend((split, path) for path in split_files)
    if not samples:
        raise FileNotFoundError(f"No synthetic NPZ models found under {dataset}")

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {
            executor.submit(
                process_sample,
                str(source),
                str(output_root / "forward_models" / split / f"{source.stem}.npz"),
                str(output_root / "migration_models" / split / f"{source.stem}.npz"),
            ): (split, source)
            for split, source in samples
        }
        for index, future in enumerate(as_completed(jobs), 1):
            split, source = jobs[future]
            try:
                row = future.result()
            except Exception as error:
                row = {"split": split, "sample_id": source.stem, "success": False,
                       "error": str(error)}
                print(f"Failed {index}/{len(samples)}: {split}/{source.stem}: {error}", flush=True)
            else:
                print(f"Processed {index}/{len(samples)}: {split}/{source.stem}; "
                      f"{row['runtime_seconds']:.2f}s", flush=True)
            rows.append(row)

    rows.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    forward_dir = output_root / "forward_models"
    forward_dir.mkdir(parents=True, exist_ok=True)
    fields = ["split", "sample_id", "synthetic_model", "forward_model",
              "migration_model", "migration_min", "migration_max",
              "runtime_seconds", "success", "error"]
    with (forward_dir / "per_sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row["success"]]
    summary = {
        "sample_count": len(samples),
        "completed_count": len(completed),
        "failed_count": len(samples) - len(completed),
        "splits": {split: sum(row["split"] == split for row in completed)
                   for split in split_directories},
        "workers": args.workers,
        "total_worker_runtime_seconds": float(
            sum(row["runtime_seconds"] for row in completed)
        ),
    }
    (forward_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
