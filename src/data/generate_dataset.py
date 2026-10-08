
"""
Generate synthetic 3D density models and surface gravity data.

Each model contains one rectangular prism with a uniform
positive density contrast.

Density model: (24, 64, 64) [z, y, x]
Gravity data:  (81, 81)     [y, x]

Density units: g/cm^3
Gravity units: mGal
Distance units: meters

Gravity is calculated using gravity_forward.py.
"""

import sys
from pathlib import Path

import numpy as np

# Project directories
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

# Allow imports from the project source folder
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gravity_migration_ml.physics.gravity_forward import calculate_gravity


# ============================================================
# 1. PARAMETERS
# ============================================================

# Density model grid
NX = 64
NY = 64
NZ = 24
CELL_SIZE = 10.0  # meters

# Prism dimensions [cells]
WIDTH_MIN = 4
WIDTH_MAX = 16
THICKNESS_MIN = 2
THICKNESS_MAX = 8

# Prism top depth [cells]
TOP_MIN = 2
TOP_MAX = 16

# Density contrast [g/cm^3]
DENSITY_MIN = 0.2
DENSITY_MAX = 1.0

# Dataset sizes
N_TRAIN = 1000
N_VAL = 100
N_TEST = 100

# Random seed for reproducibility
rng = np.random.default_rng(42)

# Output directory
OUTPUT_DIR = ROOT / "data" / "synthetic" / "single_prism"


# ============================================================
# 2. GENERATE RANDOM DENSITY MODEL
# ============================================================

def generate_density():
    """Create one random rectangular prism in a 3D density grid."""

    # Initialize empty density model
    density = np.zeros((NZ, NY, NX), dtype=np.float32)

    # Random prism dimensions [cells]
    wx = rng.integers(WIDTH_MIN, WIDTH_MAX + 1)
    wy = rng.integers(WIDTH_MIN, WIDTH_MAX + 1)
    wz = rng.integers(THICKNESS_MIN, THICKNESS_MAX + 1)

    # Random prism starting position [cells]
    # Keep at least 2 cells between prism and lateral boundaries
    x0 = rng.integers(2, NX - wx - 1)
    y0 = rng.integers(2, NY - wy - 1)
    z0 = rng.integers(TOP_MIN, min(TOP_MAX, NZ - wz) + 1)

    # Random density contrast
    rho = rng.uniform(DENSITY_MIN, DENSITY_MAX)

    # Assign density contrast to the prism
    density[
        z0:z0 + wz,
        y0:y0 + wy,
        x0:x0 + wx
    ] = rho

    # Store prism boundaries in meters
    prism = {
        "x1": int(x0 * CELL_SIZE),
        "x2": int((x0 + wx) * CELL_SIZE),
        "y1": int(y0 * CELL_SIZE),
        "y2": int((y0 + wy) * CELL_SIZE),
        "z1": int(z0 * CELL_SIZE),
        "z2": int((z0 + wz) * CELL_SIZE),
        "rho": float(rho)
    }

    return density, prism


# ============================================================
# 3. GENERATE DATASET
# ============================================================

def generate_dataset(split, n_samples):
    """Generate and save density/gravity training pairs."""

    # Create dataset folder
    folder = OUTPUT_DIR / split
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):

        # Generate random density model
        density, prism = generate_density()

        # Calculate surface gravity using gravity_forward.py
        gravity = calculate_gravity(density).astype(np.float32)

        # Save density, gravity, and prism parameters
        np.savez_compressed(
            folder / f"sample_{i:06d}.npz",
            density=density,
            gravity=gravity,
            **prism
        )

        # Print progress every 25 samples
        if (i + 1) % 25 == 0:
            print(f"{split}: {i + 1}/{n_samples}")

    print(f"Finished generating {split} dataset.")


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":

    generate_dataset("train", N_TRAIN)
    generate_dataset("validation", N_VAL)
    generate_dataset("test", N_TEST)

    print("Dataset generation complete.")
