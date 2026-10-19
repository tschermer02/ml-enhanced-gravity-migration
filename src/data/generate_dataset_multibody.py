"""Generate multi-prism synthetic density models for gravity migration ML.

Run from the repository root:
    python src/data/generate_dataset_multibody.py

Produces separate train/validation/test NPZ files containing:
    density: (24, 64, 64), g/cm^3
    n_bodies: integer
    prism_bounds_cells: (n_bodies, 6), half-open [x0,x1,y0,y1,z0,z1]
    prism_bounds_m: (n_bodies, 6), meters
    prism_densities_gcm3: (n_bodies,)

All bodies are non-overlapping rectangular prisms, separated by at least
MIN_GAP_CELLS empty cells along at least one spatial axis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Grid and physical properties: identical to the single-body dataset.
NX, NY, NZ = 64, 64, 24
CELL_SIZE_M = 10.0
WIDTH_MIN, WIDTH_MAX = 4, 16
THICKNESS_MIN, THICKNESS_MAX = 2, 8
TOP_MIN, TOP_MAX = 2, 16
DENSITY_MIN, DENSITY_MAX = 0.2, 1.0
LATERAL_MARGIN_CELLS = 2

MIN_BODIES, MAX_BODIES = 1, 3
MIN_GAP_CELLS = 2
MAX_PLACEMENT_ATTEMPTS = 1000
DEFAULT_OUTPUT_ROOT = ROOT / "src" / "data" / "dataset_multibody" / "synthetic_models"


def separated(candidate: tuple[int, ...], existing: tuple[int, ...], gap: int) -> bool:
    """True if boxes have >= gap empty cells along at least one axis."""
    ax0, ax1, ay0, ay1, az0, az1 = candidate
    bx0, bx1, by0, by1, bz0, bz1 = existing
    return (
        ax1 + gap <= bx0 or bx1 + gap <= ax0
        or ay1 + gap <= by0 or by1 + gap <= ay0
        or az1 + gap <= bz0 or bz1 + gap <= az0
    )


def random_prism(rng: np.random.Generator) -> tuple[int, ...]:
    """Sample prism bounds as half-open cell-index intervals."""
    wx = int(rng.integers(WIDTH_MIN, WIDTH_MAX + 1))
    wy = int(rng.integers(WIDTH_MIN, WIDTH_MAX + 1))
    wz = int(rng.integers(THICKNESS_MIN, THICKNESS_MAX + 1))

    # At least two empty cells at each lateral boundary.
    x0 = int(rng.integers(LATERAL_MARGIN_CELLS, NX - LATERAL_MARGIN_CELLS - wx + 1))
    y0 = int(rng.integers(LATERAL_MARGIN_CELLS, NY - LATERAL_MARGIN_CELLS - wy + 1))
    z0 = int(rng.integers(TOP_MIN, min(TOP_MAX, NZ - wz) + 1))
    return (x0, x0 + wx, y0, y0 + wy, z0, z0 + wz)


def generate_density(
    rng: np.random.Generator,
    min_bodies: int = MIN_BODIES,
    max_bodies: int = MAX_BODIES,
    gap_cells: int = MIN_GAP_CELLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return density, prism bounds (cells), and per-prism density contrasts."""
    if not 1 <= min_bodies <= max_bodies:
        raise ValueError("Require 1 <= min_bodies <= max_bodies")
    if gap_cells < 0:
        raise ValueError("gap_cells must be >= 0")

    count = int(rng.integers(min_bodies, max_bodies + 1))
    density = np.zeros((NZ, NY, NX), dtype=np.float32)
    bounds: list[tuple[int, ...]] = []
    contrasts: list[float] = []

    for body_index in range(count):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            candidate = random_prism(rng)
            if all(separated(candidate, previous, gap_cells) for previous in bounds):
                break
        else:
            raise RuntimeError(
                f"Cannot place body {body_index + 1}/{count} "
                f"after {MAX_PLACEMENT_ATTEMPTS} attempts; "
                "reduce number of bodies, prism dimensions, or gap."
            )

        rho = float(rng.uniform(DENSITY_MIN, DENSITY_MAX))
        x0, x1, y0, y1, z0, z1 = candidate
        density[z0:z1, y0:y1, x0:x1] = rho
        bounds.append(candidate)
        contrasts.append(rho)

    return (
        density,
        np.asarray(bounds, dtype=np.int32).reshape(-1, 6),
        np.asarray(contrasts, dtype=np.float32),
    )


def generate_split(
    split: str,
    sample_count: int,
    output_root: Path,
    seed: int,
    min_bodies: int,
    max_bodies: int,
    gap_cells: int,
) -> None:
    """Generate a deterministic split without overwriting existing files."""
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    folder = output_root / split
    filenames = [folder / f"sample_{i:06d}.npz" for i in range(sample_count)]
    existing = [path for path in filenames if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing files in {folder}; "
            "choose a new --output-root."
        )
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i, path in enumerate(filenames):
        density, bounds, contrasts = generate_density(
            rng, min_bodies=min_bodies, max_bodies=max_bodies, gap_cells=gap_cells
        )
        np.savez_compressed(
            path,
            density=density,
            n_bodies=np.int32(len(bounds)),
            prism_bounds_cells=bounds,
            prism_bounds_m=bounds.astype(np.float64) * CELL_SIZE_M,
            prism_densities_gcm3=contrasts,
            cell_size_m=np.float64(CELL_SIZE_M),
            seed=np.int64(seed),
        )
        if (i + 1) % 25 == 0 or i + 1 == sample_count:
            print(f"{split}: {i + 1}/{sample_count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--min-bodies", type=int, default=MIN_BODIES)
    parser.add_argument("--max-bodies", type=int, default=MAX_BODIES)
    parser.add_argument("--gap-cells", type=int, default=MIN_GAP_CELLS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if not 1 <= args.min_bodies <= args.max_bodies:
        parser.error("Require 1 <= min-bodies <= max-bodies")
    if args.gap_cells < 0:
        parser.error("--gap-cells must be >= 0")
    if min(args.train, args.validation, args.test) < 1:
        parser.error("Split sizes must be >= 1")

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    # Different deterministic random streams for each split.
    split_seeds = np.random.SeedSequence(args.seed).spawn(3)
    for split, count, child_seed in zip(
        ("train", "validation", "test"),
        (args.train, args.validation, args.test),
        split_seeds,
    ):
        seed = int(child_seed.generate_state(1, dtype=np.uint32)[0])
        generate_split(
            split, count, output_root, seed,
            args.min_bodies, args.max_bodies, args.gap_cells,
        )
    print(f"Multi-body dataset complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
