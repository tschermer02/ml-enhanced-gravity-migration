"""Generate 100 randomized held-out geology samples EACH for E05A--E05F.

All models are evaluated by the SAME pretrained E05 network; no retraining.
Density: (24,64,64) g/cm^3; gravity: (81,81) mGal.
E05F matches the E05A density for the same sample ID, adding Gaussian noise
to its gravity, enabling controlled clean-vs-noisy comparisons.

Run from repository root:
  python scripts/generate_e05a_f_cases.py --samples 100 --noise-percent 5
  python scripts/generate_e05a_f_cases.py --cases E05A E05F --samples 3
  python scripts/generate_e05a_f_cases.py --samples 100 --models-only

If files exist, rerun with --overwrite to replace them.
Physics calculation can take substantial time, particularly basement models.
"""
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

from physics.gravity_forward import calculate_gravity
from physics.gravity_migration import calculate_migration

NZ, NY, NX = 24, 64, 64
CELL_SIZE_M = 10.0
CASES = {
    "E05A": "single_compact",
    "E05B": "multiple_depths",
    "E05C": "dipping_elongated",
    "E05D": "salt_dome_like_positive",
    "E05E": "basement_relief",
    "E05F": "noisy_compact",
}
Z, Y, X = np.ogrid[:NZ, :NY, :NX]


def rng_for(seed: int, case: str, sample_index: int) -> np.random.Generator:
    """Deterministic, independent seeds; F deliberately shares A's geology."""
    stream = list(CASES).index("E05A" if case == "E05F" else case)
    return np.random.default_rng(np.random.SeedSequence([seed, stream, sample_index]))


def random_prism(rng, *, z_low=2, z_high=16):
    """Choose half-open prism intervals, with 2-cell lateral margins."""
    wx, wy = rng.integers(4, 17, size=2)
    wz = int(rng.integers(2, 9))
    x0 = int(rng.integers(2, NX - 2 - wx + 1))
    y0 = int(rng.integers(2, NY - 2 - wy + 1))
    z0 = int(rng.integers(z_low, min(z_high, NZ - wz) + 1))
    return (x0, x0 + wx, y0, y0 + wy, z0, z0 + wz)


def draw_prism(density, bounds, value):
    x0, x1, y0, y1, z0, z1 = bounds
    density[z0:z1, y0:y1, x0:x1] = value


def compact(rng):
    out = np.zeros((NZ, NY, NX), dtype=np.float32)
    draw_prism(out, random_prism(rng), float(rng.uniform(0.2, 1.0)))
    return out


def multiple_depths(rng):
    """Three non-intersecting bodies distributed between shallow and deep."""
    out = np.zeros((NZ, NY, NX), dtype=np.float32)
    # Restrict each to a disjoint vertical band, so overlap in X/Y is allowed.
    depth_bands = [(2, 7), (8, 14), (15, 22)]
    for lo, hi in depth_bands:
        wx, wy = rng.integers(4, 16, size=2)
        wz = int(rng.integers(2, min(6, hi - lo) + 1))
        x0 = int(rng.integers(2, NX - 2 - wx + 1))
        y0 = int(rng.integers(2, NY - 2 - wy + 1))
        z0 = int(rng.integers(lo, hi - wz + 1))
        draw_prism(out, (x0, x0 + wx, y0, y0 + wy, z0, z0 + wz),
                   float(rng.uniform(0.2, 1.0)))
    return out


def dipping(rng):
    """Inclined tabular feature, randomized in position, length and dip."""
    x0 = int(rng.integers(6, 17))
    length = int(rng.integers(25, 43))
    y0 = int(rng.integers(8, 43))
    width = int(rng.integers(6, 15))
    z_start = float(rng.uniform(4.0, 8.0))
    dip = float(rng.uniform(0.16, 0.34)) * int(rng.choice([-1, 1]))
    if dip < 0:
        z_start = float(rng.uniform(14.0, 19.0))
    half_thickness = float(rng.uniform(1.0, 2.2))
    mid_z = z_start + dip * (X - x0)
    mask = ((X >= x0) & (X < x0 + length)
            & (Y >= y0) & (Y < y0 + width)
            & (np.abs(Z - mid_z) <= half_thickness))
    return (mask * float(rng.uniform(0.2, 1.0))).astype(np.float32)


def dome(rng):
    """Positive-density dome analogue: randomized cap and central stem.

    Actual halite commonly has NEGATIVE density contrast, which the present
    E05 sigmoid output cannot represent.
    """
    xc = float(rng.uniform(22, 42))
    yc = float(rng.uniform(22, 42))
    radius2 = (X - xc) ** 2 + (Y - yc) ** 2
    stem_radius = float(rng.uniform(4.0, 7.0))
    stem_top = int(rng.integers(6, 10))
    cap_top = int(rng.integers(2, 6))
    center = float(rng.uniform(7.0, 10.0))
    cap_radius = (float(rng.uniform(7.0, 10.0))
                  + float(rng.uniform(2.0, 5.0))
                  * np.exp(-((Z - center) / float(rng.uniform(2.0, 3.8))) ** 2))
    stem = (radius2 <= stem_radius ** 2) & (Z >= stem_top) & (Z < 23)
    cap = (radius2 <= cap_radius ** 2) & (Z >= cap_top) & (Z <= 16)
    return ((stem | cap) * float(rng.uniform(0.2, 0.8))).astype(np.float32)


def basement(rng):
    """Variable continuous interface, denser basement below."""
    mean_depth = float(rng.uniform(12, 19))
    amp_x = float(rng.uniform(1.0, 3.0))
    amp_y = float(rng.uniform(1.0, 3.0))
    phase_x, phase_y = rng.uniform(0, 2 * np.pi, 2)
    top = (mean_depth
           + amp_x * np.sin(2 * np.pi * X / NX + phase_x)
           + amp_y * np.cos(2 * np.pi * Y / NY + phase_y))
    return ((Z >= top) * float(rng.uniform(0.2, 0.7))).astype(np.float32)


def make_density(case, seed, index):
    rng = rng_for(seed, case, index)
    if case in ("E05A", "E05F"):
        return compact(rng)
    return {
        "E05B": multiple_depths,
        "E05C": dipping,
        "E05D": dome,
        "E05E": basement,
    }[case](rng)


def check_density(density, case, index):
    if density.shape != (NZ, NY, NX):
        raise ValueError(f"{case} #{index}: density shape {density.shape}")
    if not np.isfinite(density).all() or density.min() < 0 or density.max() > 1:
        raise ValueError(f"{case} #{index}: nonfinite / out-of-range density")
    if np.count_nonzero(density) == 0:
        raise ValueError(f"{case} #{index}: empty density model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES),
                        default=list(CASES))
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of randomized samples per selected case (default 100).")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ml/evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-percent", type=float, default=5.0,
                        help="E05F: Gaussian noise std as percent of peak clean |gravity|.")
    parser.add_argument("--models-only", action="store_true",
                        help="Only write synthetic density NPZs; no forward/migration.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.noise_percent < 0:
        parser.error("--noise-percent must be >= 0")

    base = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    selected = list(dict.fromkeys(args.cases))

    # Preflight: do not partially overwrite any of the requested outputs.
    destination_paths = []
    for case in selected:
        for index in range(args.samples):
            name = f"sample_{index:06d}.npz"
            destination_paths.append(base / case / "synthetic_models" / "test" / name)
            if not args.models_only:
                destination_paths.extend([
                    base / case / "forward_models" / "test" / name,
                    base / case / "data" / "test" / name,
                ])
    existing = [p for p in destination_paths if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Found existing file {existing[0]}. "
            "Use --overwrite to regenerate; old outputs are protected."
        )

    for case in selected:
        folder = base / case
        manifest = []
        for index in range(args.samples):
            name = f"sample_{index:06d}.npz"
            density = make_density(case, args.seed, index)
            check_density(density, case, index)

            model_path = folder / "synthetic_models" / "test" / name
            model_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(model_path, density=density,
                                cell_size_m=np.float64(CELL_SIZE_M),
                                case=np.str_(case), geology=np.str_(CASES[case]),
                                sample_seed=np.int64(args.seed), sample_index=np.int32(index))
            record = {
                "sample_id": name.removesuffix(".npz"),
                "occupied_cells": int(np.count_nonzero(density)),
                "density_max_gcm3": float(density.max()),
            }
            if not args.models_only:
                clean = np.asarray(calculate_gravity(density), dtype=np.float64)
                gravity = clean.copy()
                noise_std = 0.0
                if case == "E05F":
                    noise_std = (args.noise_percent / 100) * float(np.max(np.abs(clean)))
                    # Dedicated reproducible noise RNG; does not alter underlying geology.
                    noise_rng = np.random.default_rng(
                        np.random.SeedSequence([args.seed, 999, index])
                    )
                    gravity += noise_rng.normal(0.0, noise_std, size=gravity.shape)

                migration = np.asarray(
                    calculate_migration(gravity, restrict=False), dtype=np.float64
                )
                if (gravity.shape != (81, 81) or migration.shape != (NZ, NY, NX)
                        or not np.isfinite(gravity).all()
                        or not np.isfinite(migration).all()):
                    raise ValueError(f"{case} #{index}: invalid gravity/migration")

                scale = float(np.max(np.abs(migration)))
                normalized = (migration / max(scale, 1e-8)).astype(np.float32)
                fwd = folder / "forward_models" / "test" / name
                data = folder / "data" / "test" / name
                fwd.parent.mkdir(parents=True, exist_ok=True)
                data.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    fwd, gravity=gravity.astype(np.float32),
                    gravity_clean=clean.astype(np.float32),
                    noise_std_mgal=np.float64(noise_std),
                )
                np.savez_compressed(
                    data, density=density, migration=migration.astype(np.float32),
                    migration_input=normalized, migration_scale=np.float64(scale),
                )
                record["noise_std_mgal"] = noise_std
            manifest.append(record)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.samples:
                print(f"{case}: {index+1}/{args.samples}", flush=True)

        (folder / "case_manifest.json").write_text(
            json.dumps({
                "case": case, "geology": CASES[case], "sample_count": args.samples,
                "seed": args.seed, "noise_percent": args.noise_percent if case == "E05F" else 0,
                "paired_control": "E05A" if case == "E05F" else None,
                "samples": manifest,
            }, indent=2), encoding="utf-8"
        )
    print(f"Done: {args.samples} samples in each of {len(selected)} cases; "
          f"{args.samples * len(selected)} total.", flush=True)


if __name__ == "__main__":
    main()
