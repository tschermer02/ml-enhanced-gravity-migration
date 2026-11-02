"""Balanced synthetic gravity datasets from configurable geological geometries.

Run from the repository root::

    python -m src.data.generate_dataset --name E08 --geometries compact multiple dipping salt_dome \
        --train 2000 --validation 160 --test 100 --component gz --noise 0 5 10 15

Counts mean saved samples, not underlying bodies. For paired noise, each repeated
noise condition shares the same model within a group; total counts stay fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Transitional compatibility: existing physics modules use `physics.*` imports.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SHAPE = (24, 64, 64)
NZ, NY, NX = SHAPE
CELL_M = 10.0
Z, Y, X = np.ogrid[:NZ, :NY, :NX]
GEOMETRIES = ("compact", "multiple", "multiple_depths", "dipping", "salt_dome")
SPLITS = ("train", "validation", "test")


def random_prism(rng, z_low=2, z_high=16):
    wx, wy = rng.integers(4, 17, size=2)
    wz = int(rng.integers(2, 9))
    x0 = int(rng.integers(2, NX - 2 - wx + 1))
    y0 = int(rng.integers(2, NY - 2 - wy + 1))
    z0 = int(rng.integers(z_low, min(z_high, NZ - wz) + 1))
    return x0, x0 + int(wx), y0, y0 + int(wy), z0, z0 + wz


def draw_prism(density, bounds, value):
    x0, x1, y0, y1, z0, z1 = bounds
    density[z0:z1, y0:y1, x0:x1] = value


def separated(a, b, gap):
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return (ax1 + gap <= bx0 or bx1 + gap <= ax0 or
            ay1 + gap <= by0 or by1 + gap <= ay0 or
            az1 + gap <= bz0 or bz1 + gap <= az0)


def generate_density(geometry, rng, min_bodies=1, max_bodies=3, gap_cells=2,
                     density_sign="positive"):
    """Build a density volume using the original geometry recipes.

    `multiple` uses a configurable number of separated prisms. The remaining
    modes retain the shapes of the existing research generators.

    Salt domes always have negative density contrast, regardless of
    `density_sign`. Other geometries follow the selected sign mode:
    positive, negative, or signed (random sign per sample).
    Basement geometry is not supported.
    """
    rho = np.zeros(SHAPE, dtype=np.float32)
    if geometry == "compact":
        draw_prism(rho, random_prism(rng), float(rng.uniform(.2, 1.)))
    elif geometry == "multiple":
        count = int(rng.integers(min_bodies, max_bodies + 1))
        bounds = []
        for n in range(count):
            for _ in range(1000):
                candidate = random_prism(rng)
                if all(separated(candidate, other, gap_cells) for other in bounds):
                    break
            else:
                raise RuntimeError(f"Could not place prism {n+1}; reduce body count or gap")
            draw_prism(rho, candidate, float(rng.uniform(.2, 1.)))
            bounds.append(candidate)
    elif geometry == "multiple_depths":
        for lo, hi in ((2, 7), (8, 14), (15, 22)):
            wx, wy = rng.integers(4, 16, size=2)
            wz = int(rng.integers(2, min(6, hi - lo) + 1))
            x0 = int(rng.integers(2, NX - 2 - wx + 1))
            y0 = int(rng.integers(2, NY - 2 - wy + 1))
            z0 = int(rng.integers(lo, hi - wz + 1))
            draw_prism(rho, (x0, x0 + int(wx), y0, y0 + int(wy), z0, z0 + wz),
                       float(rng.uniform(.2, 1.)))
    elif geometry == "dipping":
        x0 = int(rng.integers(6, 17))
        length = int(rng.integers(25, 43))
        y0 = int(rng.integers(8, 43))
        width = int(rng.integers(6, 15))
        z_start = float(rng.uniform(4., 8.))
        dip = float(rng.uniform(.16, .34)) * int(rng.choice([-1, 1]))
        if dip < 0:
            z_start = float(rng.uniform(14., 19.))
        thickness = float(rng.uniform(1., 2.2))
        mask = ((X >= x0) & (X < x0 + length) &
                (Y >= y0) & (Y < y0 + width) &
                (np.abs(Z - (z_start + dip * (X - x0))) <= thickness))
        rho = (mask * float(rng.uniform(.2, 1.))).astype(np.float32)
    elif geometry == "salt_dome":
        xc, yc = float(rng.uniform(22, 42)), float(rng.uniform(22, 42))
        r2 = (X - xc) ** 2 + (Y - yc) ** 2
        stem_r = float(rng.uniform(4., 7.))
        stem_top = int(rng.integers(6, 10))
        cap_top = int(rng.integers(2, 6))
        center = float(rng.uniform(7., 10.))
        cap_r = (float(rng.uniform(7., 10.)) + float(rng.uniform(2., 5.)) *
                 np.exp(-((Z - center) / float(rng.uniform(2., 3.8))) ** 2))
        mask = ((r2 <= stem_r ** 2) & (Z >= stem_top) & (Z < 23)) | \
               ((r2 <= cap_r ** 2) & (Z >= cap_top) & (Z <= 16))
        rho = (mask * float(rng.uniform(.2, .8))).astype(np.float32)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Salt domes always have negative density contrast, even when the
    # dataset was requested with --density positive.
    # All other geometries follow --density; signed selects one sign per sample.
    if geometry == "salt_dome":
        rho = -np.abs(rho)
    elif density_sign == "negative":
        rho = -np.abs(rho)
    elif density_sign == "signed":
        if rng.integers(2) == 0:
            rho = -rho
    elif density_sign != "positive":
        raise ValueError(f"Unsupported density sign mode: {density_sign}")
    if rho.shape != SHAPE or not np.isfinite(rho).all() or not np.any(rho):
        raise ValueError(f"Invalid generated density: {geometry}")
    return rho


def allocate(count, geometries, noise_levels, rng, paired_noise=False):
    """Exactly balance each geometry; noise levels distributed as evenly as possible.

    With paired_noise, each complete noise group is one unique geology model.
    Without paired_noise, each saved sample has its own underlying model.
    """
    if count % len(geometries):
        raise ValueError("Split count must be divisible by number of geometries")
    per_geometry = count // len(geometries)
    rows = []
    for gi, geometry in enumerate(geometries):
        if paired_noise:
            full, remainder = divmod(per_geometry, len(noise_levels))
            for model_id in range(full):
                for level in noise_levels:
                    rows.append((geometry, float(level), model_id))
            for j in range(remainder):
                rows.append((geometry, float(noise_levels[(gi + j) % len(noise_levels)]), full))
        else:
            for model_id in range(per_geometry):
                rows.append((geometry, float(noise_levels[(gi + model_id) % len(noise_levels)]), model_id))
    rng.shuffle(rows)
    return rows


def sample_file(root, split, row, noise_levels):
    folder = root / split / row["geometry"]
    if len(noise_levels) > 1:
        folder /= f'noise_{int(row["noise_percent"]):02d}'
    return folder / f'{row["sample_id"]}.npz'


def physics_for_sample(job):
    """Run the selected existing forward and migration operators."""
    path, component, noise, seed = job
    path = Path(path)
    with np.load(path) as arr:
        density = np.asarray(arr["density"], dtype=np.float32)
        model_id = int(arr["model_id"])
        geometry = str(arr["geometry"].item())
    from src.physics.gravity_forward import calculate_gravity
    from src.physics.gravity_migration import calculate_migration

    unit = "mGal" if component == "gz" else "E"
    clean = np.asarray(
        calculate_gravity(density, component=component),
        dtype=np.float64
    )
    std = float(noise) / 100 * float(np.max(np.abs(clean)))
    # Keep the same standard normal noise field for paired variants.
    geom_code = GEOMETRIES.index(geometry)
    rng = np.random.default_rng(np.random.SeedSequence([seed, model_id, geom_code, 991]))
    measured = clean + std * rng.standard_normal(clean.shape)
    migration = np.asarray(
         calculate_migration(measured, component=component, restrict=False),
         dtype=np.float64
    )
    scale = max(float(np.max(np.abs(migration))), 1e-8)
    if not np.isfinite(migration).all():
        raise ValueError(f"Nonfinite migration: {path}")
    # Save one self-contained sample, without requiring a second data directory.
    with np.load(path) as arr:
        initial = {key: arr[key] for key in arr.files}
    np.savez_compressed(path, **initial, gravity=measured.astype(np.float32),
                        gravity_clean=clean.astype(np.float32),
                        migration=migration.astype(np.float32),
                        migration_input=(migration / scale).astype(np.float32),
                        migration_scale=np.float64(scale), noise_std=np.float64(std),
                        observation_component=np.str_(component.upper()),
                        observation_unit=np.str_(unit))
    return str(path)


def generate_dataset(args):
    name = args.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Dataset name may contain only letters, numbers, - and _")
    root = ROOT / "outputs" / "ml" / "training" / name / "data"
    geom = list(dict.fromkeys(args.geometries))
    noise = list(dict.fromkeys(float(v) for v in args.noise))
    if not geom or not noise or min(noise) < 0 or not np.isfinite(noise).all():
        raise ValueError("Specify valid geometries and nonnegative noise levels")
    if args.min_bodies < 1 or args.max_bodies < args.min_bodies or args.gap_cells < 0:
        raise ValueError("Invalid body count or separation")
    if args.workers < 1 or min(args.train, args.validation, args.test) < 0:
        raise ValueError("Invalid split counts or workers")
    if args.train + args.validation + args.test < 1:
        raise ValueError("At least one sample is required")
    config = dict(name=name, geometries=geom, train=args.train, validation=args.validation,
                  test=args.test, component=args.component, noise=noise, paired_noise=args.paired_noise,
                  density=args.density, min_bodies=args.min_bodies,
                  max_bodies=args.max_bodies, gap_cells=args.gap_cells, seed=args.seed,
                  shape=list(SHAPE), cell_size_m=CELL_M,
                  noise_definition="Gaussian std as percent of max abs clean response",
                  layout="split/geometry[/noise_NN]/sample")
    config_path = root / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError("Existing dataset config differs; choose another --name")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Directory exists without matching config: {root}")

    # Store seeds independent of sample ordering. Paired group IDs refer to a
    # single geology model within each split (never across splits).
    rows = []
    for split_index, (split, count) in enumerate(zip(SPLITS, (args.train, args.validation, args.test))):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, split_index, 17]))
        assignment = allocate(count, geom, noise, rng, args.paired_noise)
        for idx, (geometry, level, group) in enumerate(assignment):
            sid = f"sample_{idx:06d}"
            rows.append(dict(split=split, sample_id=sid, geometry=geometry,
                             noise_percent=level, model_id=int(group)))
    if args.stage == "physics" and not config_path.exists():
        raise FileNotFoundError("Run --stage models first")
    if args.stage in ("models", "all"):
        for split_index, split in enumerate(SPLITS):
            cache = {}

            split_rows = [r for r in rows if r["split"] == split]
            total = len(split_rows)

            for i, row in enumerate(split_rows, 1):
                path = sample_file(root, split, row, noise)

                if not path.exists() or args.overwrite:
                    group_key = (row["geometry"], row["model_id"])

                    if group_key not in cache:
                        seq = np.random.SeedSequence([
                            args.seed,
                            split_index,
                            GEOMETRIES.index(row["geometry"]),
                            row["model_id"]
                        ])

                        rng = np.random.default_rng(seq)

                        cache[group_key] = generate_density(
                            row["geometry"], rng,
                            args.min_bodies, args.max_bodies,
                            args.gap_cells, args.density
                        )

                    path.parent.mkdir(parents=True, exist_ok=True)

                    np.savez_compressed(
                        path,
                        density=cache[group_key],
                        geometry=np.str_(row["geometry"]),
                        model_id=np.int32(row["model_id"]),
                        noise_percent=np.float64(row["noise_percent"])
                    )

                # Print progress every 25 samples
                if i % 25 == 0 or i == total:
                    print(f"{split}: {i}/{total}", flush=True)

            print(f"Finished generating {split} dataset.", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    with (root / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if args.stage in ("physics", "all"):
        jobs = []
        for split_index, split in enumerate(SPLITS):
            for row in (r for r in rows if r["split"] == split):
                path = sample_file(root, split, row, noise)
                if not path.exists():
                    raise FileNotFoundError(path)
                if not args.overwrite:
                    with np.load(path) as sample:
                        if "migration_input" in sample:
                            continue
                jobs.append((str(path), args.component, row["noise_percent"],
                             args.seed + split_index * 100000))
        if args.workers == 1:
            for i, job in enumerate(jobs, 1):
                physics_for_sample(job)

                if i % 25 == 0 or i == len(jobs):
                    print(f"Physics: {i}/{len(jobs)}", flush=True)

        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for i, _ in enumerate(pool.map(physics_for_sample, jobs), 1):

                    if i % 25 == 0 or i == len(jobs):
                        print(f"Physics: {i}/{len(jobs)}", flush=True)

        print("Finished calculating gravity and migration.", flush=True)
    print(f"Dataset: {root} | train={args.train}, validation={args.validation}, test={args.test}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="Experiment name (e.g., E01, E02, E07)")
    p.add_argument("--geometries", nargs="+", choices=GEOMETRIES, default=["compact"])
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--validation", type=int, default=100)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--min-bodies", type=int, default=1)
    p.add_argument("--max-bodies", type=int, default=3)
    p.add_argument("--gap-cells", type=int, default=2)
    p.add_argument("--density", choices=("positive", "negative", "signed"), default="positive",
                   help="Sign of non-salt geometries; salt domes are always negative")
    p.add_argument("--component", choices=("gz", "gzz"), default="gz")
    p.add_argument("--noise", nargs="+", type=float, default=[0.])
    p.add_argument("--paired-noise", action="store_true")
    p.add_argument("--stage", choices=("models", "physics", "all"), default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
