"""Generate six SEPARATE E05A--E05F geological challenge evaluations.

These are evaluation cases for the *existing trained E05 checkpoint*, NOT
six newly trained models. Only input geology/noise changes.

Place at scripts/generate_e05a_f_cases.py. From repository root:
    python scripts/generate_e05a_f_cases.py
    python scripts/generate_e05a_f_cases.py --cases E05A E05F --overwrite

Output for each case:
 outputs/ml/evaluation/E05A/synthetic_models/test/sample_000000.npz
 outputs/ml/evaluation/E05A/forward_models/test/sample_000000.npz
 outputs/ml/evaluation/E05A/data/test/sample_000000.npz

Dimensions and normalization match E05 exactly. E05F compares noisy gravity
against clean density truth; the gravity archive also retains clean gravity.
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
CASE_NAMES = {
    "E05A": "single_compact",
    "E05B": "multiple_depths",
    "E05C": "dipping_elongated",
    "E05D": "salt_dome_like_positive",
    "E05E": "basement_relief",
    "E05F": "noisy_gravity",
}


def make_density_models() -> dict[str, np.ndarray]:
    """Return six deterministic, positive-contrast geological density volumes."""
    z, y, x = np.ogrid[:NZ, :NY, :NX]
    cases = {}

    # A: in-distribution control, single compact rectangular prism.
    rho = np.zeros((NZ, NY, NX), dtype=np.float32)
    rho[6:12, 25:38, 23:37] = 0.65
    cases["E05A"] = rho

    # B: three disjoint prisms whose top depths vary.
    rho = np.zeros((NZ, NY, NX), dtype=np.float32)
    rho[3:7, 8:19, 10:23] = 0.45
    rho[10:16, 31:44, 29:42] = 0.70
    rho[17:22, 45:57, 7:20] = 0.85
    cases["E05B"] = rho

    # C: tabular elongated dipping feature; its z center increases with x.
    center_z = 4.0 + 0.30 * (x - 10.0)
    mask = ((x >= 10) & (x < 52) & (y >= 22) & (y < 34)
            & (np.abs(z - center_z) <= 1.6))
    cases["E05C"] = np.where(mask, 0.60, 0.0).astype(np.float32)

    # D: vertical stem + wider upper cap. Positive-density analogy only;
    # physical halite usually has negative contrast, outside E05's range.
    radius2 = (x - 32.0) ** 2 + (y - 32.0) ** 2
    stem = (radius2 <= 6.0**2) & (z >= 6) & (z <= 21)
    cap_radius = 9.0 + 4.0 * np.exp(-((z - 7.0) / 2.8) ** 2)
    cap = (radius2 <= cap_radius**2) & (z >= 3) & (z <= 16)
    cases["E05D"] = np.where(stem | cap, 0.48, 0.0).astype(np.float32)

    # E: dense material beneath a gently undulating basement interface.
    basement_top = (17.0 + 2.4 * np.sin(2 * np.pi * x / NX)
                    + 2.0 * np.cos(2 * np.pi * y / NY))
    cases["E05E"] = np.where(z >= basement_top, 0.35, 0.0).astype(np.float32)

    # F: compact prism, but the observed gravity will be corrupted by noise.
    rho = np.zeros((NZ, NY, NX), dtype=np.float32)
    rho[8:15, 18:34, 35:49] = 0.65
    cases["E05F"] = rho

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_NAMES),
                        default=list(CASE_NAMES), help="Subset, e.g. E05A E05F")
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/ml/evaluation"))
    parser.add_argument("--noise-percent", type=float, default=5.0,
                        help="Only E05F: Gaussian noise std as percent of peak |clean gravity|.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-only", action="store_true",
                        help="Only save ground-truth density models; skip physics.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.noise_percent < 0:
        parser.error("--noise-percent must be >= 0")

    base = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    models = make_density_models()
    # Deduplicate case list while respecting user order.
    selected = list(dict.fromkeys(args.cases))

    # Refuse any partial overwrite across ALL cases before writing anything.
    destinations = []
    for case in selected:
        d = base / case
        destinations.append(d / "synthetic_models/test/sample_000000.npz")
        if not args.models_only:
            destinations += [d / "forward_models/test/sample_000000.npz",
                             d / "data/test/sample_000000.npz"]
    already = [p for p in destinations if p.exists()]
    if already and not args.overwrite:
        raise FileExistsError(f"{already[0]} already exists; pass --overwrite to regenerate.")

    rng = np.random.default_rng(args.seed)
    for i, case in enumerate(selected, 1):
        density = models[case]
        if (density.shape != (NZ, NY, NX) or not np.isfinite(density).all()
                or np.min(density) < 0 or np.max(density) > 1):
            raise ValueError(f"Invalid density for {case}: {density.shape}")
        folder = base / case
        density_path = folder / "synthetic_models/test/sample_000000.npz"
        density_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(density_path, density=density,
                            cell_size_m=np.float64(CELL_SIZE_M),
                            case=np.str_(case), geology=np.str_(CASE_NAMES[case]))
        meta = {"case": case, "geology": CASE_NAMES[case],
                "density_shape": list(density.shape),
                "occupied_cells": int(np.count_nonzero(density)),
                "density_max_gcm3": float(density.max())}

        if not args.models_only:
            clean = np.asarray(calculate_gravity(density), dtype=np.float64)
            observed = clean.copy()
            noise_std = 0.0
            if case == "E05F":
                noise_std = (args.noise_percent / 100.0) * float(np.max(np.abs(clean)))
                observed += rng.normal(loc=0.0, scale=noise_std, size=clean.shape)
            migration = np.asarray(calculate_migration(observed, restrict=False),
                                   dtype=np.float64)
            if observed.shape != (81, 81) or migration.shape != (NZ, NY, NX):
                raise ValueError(f"Unexpected forward/migration shapes for {case}")
            if not np.isfinite(observed).all() or not np.isfinite(migration).all():
                raise ValueError(f"Non-finite physics result for {case}")
            scale = float(np.max(np.abs(migration)))
            normalized = (migration / max(scale, 1e-8)).astype(np.float32)

            fwd_path = folder / "forward_models/test/sample_000000.npz"
            input_path = folder / "data/test/sample_000000.npz"
            fwd_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fwd_path, gravity=observed.astype(np.float32),
                                gravity_clean=clean.astype(np.float32),
                                noise_std_mgal=np.float64(noise_std))
            np.savez_compressed(input_path, density=density,
                                migration=migration.astype(np.float32),
                                migration_input=normalized,
                                migration_scale=np.float64(scale))
            meta.update(noise_std_mgal=noise_std, migration_scale=scale)

        (folder / "case_manifest.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[{i}/{len(selected)}] {case} ({CASE_NAMES[case]}): "
              f"{meta['occupied_cells']} occupied cells", flush=True)
    print(f"Done. Evaluation inputs under {base}", flush=True)


if __name__ == "__main__":
    main()
