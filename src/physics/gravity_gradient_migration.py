"""MATLAB gg_migration2014.m, migid=7: depth-weighted Gzz migration.

The MATLAB implementation uses Gzz data in Eotvos, its case-7 kernel,
and multiplier zwm^2 / (gamma*sqrt(3*pi)/2), with gamma=6.67e-11*1e8.
Here flat receivers z=0 and zero clearance imply zwm = depth.
This is MATLAB's weighted migration, NOT an unweighted adjoint A.T @ data.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from physics.gravity_gradient_forward import (G, CELL_SIZE, SHAPE, OBS_SHAPE, gzz_kernel)

GAMMA = G * 1e8


def calculate_gzz_migration(gravity: np.ndarray, restrict: bool = False,
                             ro_min: float = -1.0, ro_max: float = 1.0) -> np.ndarray:
    """Compute MATLAB Gzz migration for a flat survey; input E, output proxy volume.

    MATLAB's optional `restrictid` linear maps the volume to [ro_min,ro_max].
    Use restrict=False for signed CNN migration normalized by per-sample max abs.
    """
    gravity = np.asarray(gravity, dtype=np.float64)
    if gravity.shape != OBS_SHAPE or not np.isfinite(gravity).all():
        raise ValueError(f'Gzz must be finite with shape {OBS_SHAPE}')
    if ro_min > ro_max:
        raise ValueError('ro_min > ro_max')
    volume = np.empty(SHAPE, dtype=np.float64)
    czz = GAMMA * np.sqrt(3 * np.pi) / 2.0
    for iz in range(SHAPE[0]):
        depth = (iz + .5) * CELL_SIZE
        kernel = gzz_kernel(depth)
        full = fftconvolve(gravity, kernel[::-1, ::-1], mode='full')
        volume[iz] = full[80:80+SHAPE[1], 80:80+SHAPE[2]] * depth**2 / czz
    if restrict:
        lo, hi = float(volume.min()), float(volume.max())
        if hi > lo:
            volume = (volume-lo) * ((ro_max-ro_min)/(hi-lo)) + ro_min
        else:
            volume.fill(ro_min)
    return volume


calculate_migration = calculate_gzz_migration
