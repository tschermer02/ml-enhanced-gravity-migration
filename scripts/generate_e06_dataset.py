"""Build a mixed-geology E06 training/validation/test dataset.

Uses the SAME model shapes and forward/migration normalization as E05.
The six geology categories are imported unchanged from generate_e05a_f_cases.py.
Default: 2000 train, 120 validation, 22 held-out samples EACH of six
categories (132 test). Training/validation are balanced across categories.

Run at repository root:
  python scripts/generate_e06_dataset.py --stage models
  python scripts/generate_e06_dataset.py --stage physics --workers 4
Or both at once:
  python scripts/generate_e06_dataset.py --stage all --workers 4

CAUTION: forward modeling thousands of dense basement volumes is costly;
run the physics stage in a CHPC compute allocation (not login node).
Existing output files are skipped by default, which permits resuming.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (ROOT, SRC):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from generate_e05a_f_cases import CASES, make_density, check_density
from physics.gravity_forward import calculate_gravity
from physics.gravity_migration import calculate_migration

CASE_NAMES = tuple(CASES)
SPLIT_SEEDS = {"train": 1001, "validation": 2002, "test": 3003}
# Keep F's positive, compact target identical to the corresponding A target.
# Noise only affects F's GRAVITY observation, never the density ground truth.


def records_for_split(split: str, count: int, seed: int) -> list[dict]:
    """Round-robin geology distribution, independently shuffled by split."""
    labels = [CASE_NAMES[i % len(CASE_NAMES)] for i in range(count)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_SEEDS[split], 54321]))
    rng.shuffle(labels)
    indices = {case: 0 for case in CASE_NAMES}
    records = []
    for sample_id, case in enumerate(labels):
        local_index = indices[case]
        indices[case] += 1
        records.append({"sample_id": f"sample_{sample_id:06d}", "case": case,
                        "case_index": local_index, "split": split})
    return records


def generate_models(dataset_root: Path, manifest_path: Path, counts: dict,
                    seed: int, overwrite: bool) -> list[dict]:
    """Write all six kinds of ground-truth density with split-isolated seeds."""
    all_records = []
    for split, count in counts.items():
        records = records_for_split(split, count, seed)
        for idx, record in enumerate(records, 1):
            path = dataset_root / "synthetic_models" / split / (record["sample_id"] + ".npz")
            if path.exists() and not overwrite:
                # Always check exact metadata on resume, never silently mix runs.
                with np.load(path) as archived:
                    saved_case = str(archived["case"].item())
                    saved_idx = int(archived["case_index"])
                if saved_case != record["case"] or saved_idx != record["case_index"]:
                    raise ValueError(f"Existing dataset differs from requested run: {path}")
            else:
                density = make_density(record["case"],
                                       seed + SPLIT_SEEDS[split], record["case_index"])
                check_density(density, record["case"], record["case_index"])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, density=density, case=np.str_(record["case"]),
                                    case_index=np.int32(record["case_index"]),
                                    split=np.str_(split), seed=np.int64(seed),
                                    cell_size_m=np.float64(10))
            all_records.append(record)
            if idx % 100 == 0 or idx == count:
                print(f"Density models {split}: {idx}/{count}", flush=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "case", "case_index", "split"])
        writer.writeheader()
        writer.writerows(all_records)
    return all_records


def run_physics_one(job: tuple) -> str:
    """Forward gravity, optional measurement noise, migration, and normalization."""
    source, gravity_path, data_path, seed, split_offset, percent, overwrite = job
    source, gravity_path, data_path = map(Path, (source, gravity_path, data_path))
    if gravity_path.is_file() and data_path.is_file() and not overwrite:
        return "skipped"
    with np.load(source) as f:
        density = np.asarray(f["density"], dtype=np.float32)
        case = str(f["case"].item())
        case_index = int(f["case_index"])
    clean = np.asarray(calculate_gravity(density), dtype=np.float64)
    measured = clean.copy()
    noise_std = 0.0
    if case == "E05F":
        noise_std = percent / 100.0 * float(np.max(np.abs(clean)))
        noise_rng = np.random.default_rng(np.random.SeedSequence([
            seed, split_offset, 999, case_index
        ]))
        measured += noise_rng.normal(0.0, noise_std, size=measured.shape)
    migration = np.asarray(calculate_migration(measured, restrict=False), dtype=np.float64)
    scale = float(np.max(np.abs(migration)))
    normalized = (migration / max(scale, 1e-8)).astype(np.float32)
    if not (np.isfinite(measured).all() and np.isfinite(normalized).all()):
        raise ValueError(f"Nonfinite data: {source}")
    gravity_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gravity_path, gravity=measured.astype(np.float32),
                        gravity_clean=clean.astype(np.float32),
                        noise_std_mgal=np.float64(noise_std))
    np.savez_compressed(data_path, density=density,
                        migration=migration.astype(np.float32),
                        migration_input=normalized, migration_scale=np.float64(scale))
    return "created"


def generate_physics(dataset_root: Path, prepared_root: Path, records: list[dict],
                     seed: int, noise_percent: float, workers: int, overwrite: bool):
    jobs = []
    for rec in records:
        name = rec["sample_id"] + ".npz"
        split = rec["split"]
        source = dataset_root / "synthetic_models" / split / name
        if not source.is_file():
            raise FileNotFoundError(f"Generate models first: {source}")
        jobs.append((str(source), str(dataset_root / "forward_models" / split / name),
                     str(prepared_root / split / name), seed, SPLIT_SEEDS[split],
                     noise_percent, overwrite))
    # Explicit bounded worker count: forward calls can use significant memory.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_physics_one, job): job[0] for job in jobs}
        done = 0
        for future in as_completed(futures):
            result = future.result()  # Propagate failures; never silently lose samples.
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"Forward + migration {done}/{len(jobs)} ({result})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "models", "physics"), default="all")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--validation", type=int, default=120)
    p.add_argument("--test-per-case", type=int, default=22)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-percent", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dataset-root", type=Path, default=Path("src/data/dataset_e06"))
    p.add_argument("--data-root", type=Path, default=Path("outputs/ml/training/E06/data"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if min(args.train, args.validation, args.test_per_case, args.workers) < 1:
        p.error("train, validation, test-per-case and workers must be >= 1")
    if args.noise_percent < 0:
        p.error("noise-percent must be >= 0")
    if args.train < 6 or args.validation < 6:
        p.error("train/validation must contain at least one of each geology type")

    data_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    prepared_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    manifest_path = data_root / "manifest.csv"
    counts = {"train": args.train, "validation": args.validation,
              "test": args.test_per_case * len(CASE_NAMES)}
    if args.stage in ("all", "models"):
        records = generate_models(data_root, manifest_path, counts, args.seed, args.overwrite)
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}; run --stage models")
        with manifest_path.open(newline="", encoding="utf-8") as file:
            records = list(csv.DictReader(file))
        for split, count in counts.items():
            if sum(r["split"] == split for r in records) != count:
                raise ValueError(f"Manifest {split} count does not match requested {count}")

    if args.stage in ("all", "physics"):
        generate_physics(data_root, prepared_root, records, args.seed,
                         args.noise_percent, args.workers, args.overwrite)
    config = {"experiment": "E06", "seed": args.seed, "splits": counts,
              "test_per_case": args.test_per_case, "geology_types": CASES,
              "noise_percent_for_E05F": args.noise_percent,
              "target": "positive density contrast, g/cm^3",
              "normalization": "per-sample max absolute gravity migration"}
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "dataset_config.json").write_text(json.dumps(config, indent=2))
    print(f"E06 {args.stage} done. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
