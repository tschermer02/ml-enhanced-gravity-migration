"""Cell-centered gravity forward models on the project's regular survey grid.

Input density (24, 64, 64) in g/cm^3. Gz returns mGal, Gzz returns Eotvos.
Coordinates use z positive downward. Numerical conventions match legacy code.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

G = 6.67e-11
CELL_SIZE = 10.0
SHAPE = (24, 64, 64)
OBS_SHAPE = (81, 81)
NZ, NY, NX = SHAPE
OBS_X = np.linspace(-85, 715, OBS_SHAPE[1])
OBS_Y = np.linspace(-85, 715, OBS_SHAPE[0])
OBS_Z = 0.0
BATCH_SIZE = 128
_OFFSETS = np.arange(-72, 72, dtype=np.float64) * CELL_SIZE
_DY, _DX = np.meshgrid(_OFFSETS, _OFFSETS, indexing="ij")


def _density_array(density):
    density = np.asarray(density, dtype=np.float64)
    if density.shape != SHAPE or not np.isfinite(density).all():
        raise ValueError(f"Density must be finite with shape {SHAPE}")
    return density


def calculate_gz(density):
    """Compute vertical gravity Gz using the original point-mass sum (mGal)."""
    density = _density_array(density)
    iz, iy, ix = np.nonzero(density)
    if not len(iz):
        return np.zeros(OBS_SHAPE, dtype=np.float64)
    x, y, z = (ix + .5) * CELL_SIZE, (iy + .5) * CELL_SIZE, (iz + .5) * CELL_SIZE
    mass = G * density[iz, iy, ix] * CELL_SIZE**3 * 1e8
    xx, yy = np.meshgrid(OBS_X, OBS_Y)
    ox, oy = xx.ravel(), yy.ravel()
    values = np.empty(ox.size, dtype=np.float64)
    for start in range(0, ox.size, BATCH_SIZE):
        end = min(start + BATCH_SIZE, ox.size)
        dx = x[None, :] - ox[start:end, None]
        dy = y[None, :] - oy[start:end, None]
        dz = z[None, :] - OBS_Z
        r2 = dx**2 + dy**2 + dz**2
        values[start:end] = np.sum(mass[None, :] * dz / (r2 * np.sqrt(r2)), axis=1)
    return values.reshape(OBS_SHAPE)


def gzz_kernel(depth_m):
    """Spatial Gzz kernel (3*z^2-r^2)/r^5 used by the MATLAB operator."""
    r2 = _DX**2 + _DY**2 + float(depth_m)**2
    return (3.0 * float(depth_m)**2 - r2) / (r2**2 * np.sqrt(r2))


def calculate_gzz(density):
    """Compute Gzz using lateral FFT convolution (Eotvos)."""
    density = _density_array(density)
    result = np.zeros(OBS_SHAPE, dtype=np.float64)
    multiplier = G * CELL_SIZE**3 * 1e12
    for iz, layer in enumerate(density):
        if not np.any(layer):
            continue
        full = fftconvolve(layer, gzz_kernel((iz + .5) * CELL_SIZE), mode="full")
        result += full[63:63 + OBS_SHAPE[0], 63:63 + OBS_SHAPE[1]] * multiplier
    return result


def calculate_gravity(density, component="gz"):
    """Select the forward model with component='gz' or 'gzz'."""
    component = component.lower()
    if component == "gz":
        return calculate_gz(density)
    if component == "gzz":
        return calculate_gzz(density)
    raise ValueError("component must be 'gz' or 'gzz'")
