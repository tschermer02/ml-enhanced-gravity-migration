"""MATLAB getPredGR.m case 7: Gzz forward response (Eotvos).

Cell-centered pulse basis, 10 m cubes, shape (24,64,64), observation (81,81).
Coordinates z-positive-down, observations z=0. A voxel has density contrast in
 g/cm^3; outputs E (1 E = 1e-9 s^-2). Uses FFT lateral convolution, equivalent
 to the MATLAB cell-centered kernel on this translation-invariant regular grid.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

G = 6.67e-11  # match MATLAB
CELL_SIZE = 10.0
SHAPE = (24, 64, 64)
OBS_SHAPE = (81, 81)
OFFSETS = np.arange(-72, 72, dtype=np.float64) * CELL_SIZE
DY, DX = np.meshgrid(OFFSETS, OFFSETS, indexing='ij')


def gzz_kernel(depth_m: float) -> np.ndarray:
    """Dimensionless spatial part: (2 dz² - dx² - dy²) / r^5."""
    r2 = DX**2 + DY**2 + float(depth_m)**2
    return (3.0 * float(depth_m)**2 - r2) / (r2**2 * np.sqrt(r2))


def calculate_gzz(density: np.ndarray) -> np.ndarray:
    """Compute MATLAB-compatible Gzz, in Eotvos, with 1e4 gradient conversion.

    gamma=1e8*G*w in MATLAB, then 1e4 for Gzz -> 1e12 G*w.
    w here is volume CELL_SIZE**3, density is g/cm^3.
    """
    density = np.asarray(density, dtype=np.float64)
    if density.shape != SHAPE or not np.isfinite(density).all():
        raise ValueError(f'density must be finite with shape {SHAPE}')
    obs = np.zeros(OBS_SHAPE, dtype=np.float64)
    multiplier = G * CELL_SIZE**3 * 1e12
    for iz, layer in enumerate(density):
        if not np.any(layer):
            continue
        kernel = gzz_kernel((iz + .5) * CELL_SIZE)
        full = fftconvolve(layer, kernel, mode='full')
        # Kernel indices (-72..71) correspond to obs-source offsets;
        # source cell centers = 5+10i, receiver centers = -85+10j.
        # Thus obs-source = 10(j-i-9), and full-convolution index j+63.
        obs += full[63:63+OBS_SHAPE[0], 63:63+OBS_SHAPE[1]] * multiplier
    return obs


# alias for convenient migration pipeline
calculate_gravity = calculate_gzz
