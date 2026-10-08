
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

    # --------------------------------------------------------
    # 2.2 Create observation coordinates
    # --------------------------------------------------------

    xr, yr = np.meshgrid(OBS_X, OBS_Y)

    xr = xr.ravel()
    yr = yr.ravel()
    zr = np.full(xr.size, OBS_Z)

    # Flatten gravity measurements in matching order
    gz = gravity.ravel()

    # --------------------------------------------------------
    # 2.3 Create subsurface model coordinates
    # --------------------------------------------------------

    # Use the centers of model cells
    x = (np.arange(NX) + 0.5) * CELL_SIZE
    y = (np.arange(NY) + 0.5) * CELL_SIZE
    z = (np.arange(NZ) + 0.5) * CELL_SIZE

    # 3D coordinate grids in [z, y, x] order
    zz, yy, xx = np.meshgrid(
        z, y, x,
        indexing="ij"
    )

    # Flatten model coordinates for calculation
    xm = xx.ravel()
    ym = yy.ravel()
    zm = zz.ravel()

    # --------------------------------------------------------
    # 2.4 Calculate MATLAB migration summation
    # --------------------------------------------------------

    dm = np.zeros(xm.size, dtype=np.float64)

    # Process model cells in batches to reduce memory usage
    for start in range(0, xm.size, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, xm.size)

        # Coordinate differences between receivers and cells
        dx = xr[None, :] - xm[start:end, None]
        dy = yr[None, :] - ym[start:end, None]
        dz = zm[start:end, None] - zr[None, :]

        # Squared distance
        r2 = dx**2 + dy**2 + dz**2

        # Distance cubed
        r3 = r2 * np.sqrt(r2)

        # MATLAB Gz migration kernel:
        #
        # gzm = vr(ir) .* zzd ./ r3
        # dm = dm + gzm
        dm[start:end] = np.sum(
            gz[None, :] * dz / r3,
            axis=1
        )

    # --------------------------------------------------------
    # 2.5 Apply MATLAB depth weighting
    # --------------------------------------------------------

    # MATLAB:
    # cz = gamma * sqrt(pi)
    # dm = abs(zwm) .* dm ./ cz
    #
    # With flat topography and zero receiver clearance:
    # zwm = zm

    cz = GAMMA * np.sqrt(np.pi)

    dm = np.abs(zm) * dm / cz

    # --------------------------------------------------------
    # 2.6 Optional MATLAB range restriction
    # --------------------------------------------------------

    if restrict:

        dm_min = dm.min()
        dm_max = dm.max()

        # Avoid division by zero if all values are equal
        if dm_max > dm_min:
            zoom = (ro_max - ro_min) / (dm_max - dm_min)
            dm = (dm - dm_min) * zoom + ro_min
        else:
            dm[:] = ro_min

    # --------------------------------------------------------
    # 2.7 Reshape into 3D model
    # --------------------------------------------------------

    migration = dm.reshape(NZ, NY, NX)

    return migration
