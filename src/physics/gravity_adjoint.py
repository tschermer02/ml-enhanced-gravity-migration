"""Two distinct adjoint-like operators for Gz.

The surface-integral adjoint is not the discrete transpose of the forward map;
keep their normalization factors distinct for scientific interpretation.
"""
import numpy as np
from .gravity_forward import G, CELL_SIZE, SHAPE, OBS_SHAPE, OBS_X, OBS_Y, OBS_Z

CELL_BATCH = 64


def _backproject(gravity):
    gravity = np.asarray(gravity, dtype=np.float64)
    if gravity.shape != OBS_SHAPE or not np.isfinite(gravity).all():
        raise ValueError(f"Gravity must be finite with shape {OBS_SHAPE}")
    ox, oy = np.meshgrid(OBS_X, OBS_Y)
    ox, oy = ox.ravel(), oy.ravel()
    iz, iy, ix = np.indices(SHAPE)
    x, y, z = (ix.ravel() + .5) * CELL_SIZE, (iy.ravel() + .5) * CELL_SIZE, (iz.ravel() + .5) * CELL_SIZE
    result = np.empty(x.size, dtype=np.float64)
    data = gravity.ravel()
    for start in range(0, x.size, CELL_BATCH):
        stop = min(start + CELL_BATCH, x.size)
        dx = x[start:stop, None] - ox[None, :]
        dy = y[start:stop, None] - oy[None, :]
        dz = z[start:stop, None] - OBS_Z
        r2 = dx**2 + dy**2 + dz**2
        result[start:stop] = np.sum(data[None, :] * dz / (r2 * np.sqrt(r2)), axis=1)
    return result.reshape(SHAPE)


def calculate_adjoint(gravity):
    """Surface-integral adjoint image (not a density reconstruction)."""
    return G * (OBS_X[1] - OBS_X[0]) * (OBS_Y[1] - OBS_Y[0]) * _backproject(gravity)


def calculate_discrete_transpose(gravity):
    """Discrete transpose of the cell-centered Gz forward operator."""
    return G * 1e8 * CELL_SIZE**3 * _backproject(gravity)
