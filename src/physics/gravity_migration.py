
"""
3D gravity migration using the 2014 MATLAB implementation.

Reproduces the Gz migration kernel and depth weighting
from gg_migration2014.m.

Input:
    Gravity: (81, 81) [y, x] in mGal

Output:
    Migration: (24, 64, 64) [z, y, x]

Coordinates:
    x = east
    y = north
    z = positive downward

Assumes flat topography and zero receiver clearance.
"""

import numpy as np
from scipy.signal import fftconvolve


# ============================================================
# 1. PARAMETERS
# ============================================================

# Gravitational constant with MATLAB unit conversion
GAMMA = 6.67e-11 * 1e8

# Model grid
NX = 64
NY = 64
NZ = 24
CELL_SIZE = 10.0

# Surface gravity observation grid
OBS_X = np.linspace(-85, 715, 81)
OBS_Y = np.linspace(-85, 715, 81)
OBS_Z = 0.0

# Number of model cells processed at once
CHUNK_SIZE = 256


# ============================================================
# 2. GRAVITY MIGRATION
# ============================================================

def calculate_migration(gravity, restrict=False,
                        ro_min=0.0, ro_max=1.0):
    """
    Calculate 3D gravity migration from surface Gz.

    Uses the original MATLAB Gz migration equation:

        dm = sum(Gz * dz / r^3)

        dm = abs(z) * dm / (gamma * sqrt(pi))

    Parameters
    ----------
    gravity : np.ndarray
        Surface Gz measurements, shape (81, 81), in mGal.

    restrict : bool
        If True, rescale migration values to [ro_min, ro_max].

    ro_min, ro_max : float
        Minimum and maximum values for optional rescaling.

    Returns
    -------
    np.ndarray
        3D migration image, shape (24, 64, 64).
    """

    # --------------------------------------------------------
    # 2.1 Validate gravity data
    # --------------------------------------------------------

    gravity = np.asarray(gravity, dtype=np.float64)

    if gravity.shape != (len(OBS_Y), len(OBS_X)):
        raise ValueError("Gravity must have shape (81, 81)")

    if not np.isfinite(gravity).all():
        raise ValueError("Gravity contains NaN or infinity")

    if ro_min > ro_max:
        raise ValueError("ro_min must not exceed ro_max")

    offsets = np.arange(-72, 72, dtype=np.float64) * CELL_SIZE
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    migration = np.empty((NZ, NY, NX), dtype=np.float64)
    normalization = GAMMA * np.sqrt(np.pi)

    for iz in range(NZ):
        depth = (iz + 0.5) * CELL_SIZE
        radius_squared = dx**2 + dy**2 + depth**2
        kernel = depth / (radius_squared * np.sqrt(radius_squared))
        convolution = fftconvolve(gravity, kernel[::-1, ::-1], mode="full")
        migration[iz] = (
            convolution[80:80 + NY, 80:80 + NX] * depth / normalization
        )

    if restrict:
        migration_min = migration.min()
        migration_max = migration.max()
        if migration_max > migration_min:
            scale = (ro_max - ro_min) / (migration_max - migration_min)
            migration = (migration - migration_min) * scale + ro_min
        else:
            migration[:] = ro_min

    return migration
