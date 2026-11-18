"""MATLAB-style depth-weighted migration for Gz (mGal) and Gzz (Eotvos).

This is not the unweighted exact discrete transpose of the forward operator.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from .gravity_forward import (
    G,
    CELL_SIZE,
    SHAPE,
    OBS_SHAPE,
    gzz_kernel,
    E11_DX_M,
    E11_DY_M,
    E11_DZ_M,
)


def calculate_migration(
    gravity,
    component="gz",
    restrict=False,
    ro_min=None,
    ro_max=None,
    dx_m=CELL_SIZE,
    dy_m=CELL_SIZE,
    dz_m=CELL_SIZE,
    observation_height_m=0.0,
):
    """Migrate a surface anomaly onto (z,y,x) voxel grid.

    For Gz, default restricted range is [0,1]; for Gzz, [-1,1].
    Set restrict=False to preserve raw values for per-sample normalization.
    """
    component = component.lower()
    if component not in ("gz", "gzz"):
        raise ValueError("component must be 'gz' or 'gzz'")
    if not np.all(np.isfinite([dx_m, dy_m, dz_m, observation_height_m])):
        raise ValueError("Cell dimensions and observation height must be finite")

    if min(dx_m, dy_m, dz_m) <= 0:
        raise ValueError("Cell dimensions must be positive")

    if observation_height_m < 0:
        raise ValueError("Observation height must be nonnegative")
    
    gravity = np.asarray(gravity, dtype=np.float64)
    if gravity.shape != OBS_SHAPE or not np.isfinite(gravity).all():
        raise ValueError(f"Gravity must be finite with shape {OBS_SHAPE}")
    if ro_min is None: ro_min = 0.0 if component == "gz" else -1.0
    if ro_max is None: ro_max = 1.0
    if ro_min > ro_max:
        raise ValueError("ro_min must not exceed ro_max")
    gamma = G * 1e8

    # Horizontal offsets between observations and model cells
    offsets_x = np.arange(-72, 72, dtype=np.float64) * dx_m
    offsets_y = np.arange(-72, 72, dtype=np.float64) * dy_m

    dy, dx = np.meshgrid(
        offsets_y,
        offsets_x,
        indexing="ij"
    )

    result = np.empty(SHAPE, dtype=np.float64)

    for iz in range(SHAPE[0]):

        # Physical depth of the cell center
        depth = (iz + 0.5) * dz_m

        # Distance from the cell center to the elevated observation plane
        separation = depth + observation_height_m

        if component == "gz":

            r2 = dx**2 + dy**2 + separation**2

            kernel = separation / (r2 * np.sqrt(r2))

            multiplier = (
                separation /
                (gamma * np.sqrt(np.pi))
            )

        else:

            kernel = gzz_kernel(
                separation,
                dx_m=dx_m,
                dy_m=dy_m
            )

            multiplier = (
                separation**2 /
                (gamma * np.sqrt(3 * np.pi) / 2.0)
            )

        full = fftconvolve(
            gravity,
            kernel[::-1, ::-1],
            mode="full"
        )

        result[iz] = (
            full[
                80:80 + SHAPE[1],
                80:80 + SHAPE[2]
            ] * multiplier
        )
    if restrict:
        lo, hi = result.min(), result.max()
        if hi > lo:
            result = (result - lo) * ((ro_max - ro_min) / (hi - lo)) + ro_min
        else:
            result.fill(ro_min)
    return result


def calculate_gzz_migration(
    gravity,
    restrict=False,
    ro_min=-1.0,
    ro_max=1.0,
    dx_m=CELL_SIZE,
    dy_m=CELL_SIZE,
    dz_m=CELL_SIZE,
    observation_height_m=0.0,
):
    """Convenience alias for the Gzz migration operator."""

    return calculate_migration(
        gravity,
        component="gzz",
        restrict=restrict,
        ro_min=ro_min,
        ro_max=ro_max,
        dx_m=dx_m,
        dy_m=dy_m,
        dz_m=dz_m,
        observation_height_m=observation_height_m,
    )


def normalize_migration(migration, epsilon=1e-8):
    """Return max-absolute normalized migration and its unmodified scale."""
    migration = np.asarray(migration, dtype=np.float64)
    if not np.isfinite(migration).all():
        raise ValueError("Migration contains nonfinite values")
    scale = float(np.max(np.abs(migration)))
    return (migration / max(scale, epsilon)).astype(np.float32), scale
