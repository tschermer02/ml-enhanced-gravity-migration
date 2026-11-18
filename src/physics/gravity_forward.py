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

# E11 regional model geometry
E11_X_EXTENT_M = 8000.0
E11_Y_EXTENT_M = 8000.0
E11_Z_EXTENT_M = 1000.0

E11_DX_M = E11_X_EXTENT_M / NX
E11_DY_M = E11_Y_EXTENT_M / NY
E11_DZ_M = E11_Z_EXTENT_M / NZ


OBS_X = np.linspace(-85, 715, OBS_SHAPE[1])
OBS_Y = np.linspace(-85, 715, OBS_SHAPE[0])
OBS_Z = 0.0

def observation_coordinates(dx_m=CELL_SIZE, dy_m=CELL_SIZE):
    """Return observation grid coordinates for the requested cell sizes."""

    obs_x = (np.arange(OBS_SHAPE[1]) - 8.5) * dx_m
    obs_y = (np.arange(OBS_SHAPE[0]) - 8.5) * dy_m

    return obs_x, obs_y

BATCH_SIZE = 128
_OFFSETS = np.arange(-72, 72, dtype=np.float64) * CELL_SIZE
_DY, _DX = np.meshgrid(_OFFSETS, _OFFSETS, indexing="ij")


def _density_array(density):
    density = np.asarray(density, dtype=np.float64)
    if density.shape != SHAPE or not np.isfinite(density).all():
        raise ValueError(f"Density must be finite with shape {SHAPE}")
    return density


def calculate_gz(
    density,
    dx_m=CELL_SIZE,
    dy_m=CELL_SIZE,
    dz_m=CELL_SIZE,
    observation_height_m=0.0
):
    """Compute vertical gravity Gz using the point-mass sum (mGal)."""

    density = _density_array(density)

    if min(dx_m, dy_m, dz_m) <= 0:
        raise ValueError("Cell dimensions must be positive")

    if observation_height_m < 0:
        raise ValueError("Observation height must be nonnegative")

    iz, iy, ix = np.nonzero(density)

    if not len(iz):
        return np.zeros(OBS_SHAPE, dtype=np.float64)

    # Physical cell-center coordinates
    x = (ix + 0.5) * dx_m
    y = (iy + 0.5) * dy_m
    z = (iz + 0.5) * dz_m

    # Density g/cm^3 -> kg/m^3, gravity SI -> mGal
    cell_volume = dx_m * dy_m * dz_m
    mass = G * density[iz, iy, ix] * cell_volume * 1e8

    obs_x, obs_y = observation_coordinates(dx_m, dy_m)

    xx, yy = np.meshgrid(obs_x, obs_y)
    ox, oy = xx.ravel(), yy.ravel()

    values = np.empty(ox.size, dtype=np.float64)

    for start in range(0, ox.size, BATCH_SIZE):

        end = min(start + BATCH_SIZE, ox.size)

        delta_x = x[None, :] - ox[start:end, None]
        delta_y = y[None, :] - oy[start:end, None]
        delta_z = z[None, :] + observation_height_m

        r2 = delta_x**2 + delta_y**2 + delta_z**2

        values[start:end] = np.sum(
            mass[None, :] * delta_z /
            (r2 * np.sqrt(r2)),
            axis=1
        )

    return values.reshape(OBS_SHAPE)

def gzz_kernel(depth_m, dx_m=CELL_SIZE, dy_m=CELL_SIZE):
    """Spatial Gzz kernel supporting different horizontal cell sizes."""

    offsets_x = np.arange(-72, 72, dtype=np.float64) * dx_m
    offsets_y = np.arange(-72, 72, dtype=np.float64) * dy_m

    dy, dx = np.meshgrid(
        offsets_y,
        offsets_x,
        indexing="ij"
    )

    r2 = dx**2 + dy**2 + float(depth_m)**2

    return (3.0 * float(depth_m)**2 - r2) / (r2**2 * np.sqrt(r2))


def calculate_gzz(
    density,
    dx_m=CELL_SIZE,
    dy_m=CELL_SIZE,
    dz_m=CELL_SIZE,
    observation_height_m=0.0
):
    """Compute Gzz using lateral FFT convolution (Eotvos)."""

    density = _density_array(density)

    if min(dx_m, dy_m, dz_m) <= 0:
        raise ValueError("Cell dimensions must be positive")

    if observation_height_m < 0:
        raise ValueError("Observation height must be nonnegative")

    result = np.zeros(OBS_SHAPE, dtype=np.float64)

    # Physical volume of each model cell
    cell_volume = dx_m * dy_m * dz_m

    # Density in g/cm^3 and output in Eotvos
    multiplier = G * cell_volume * 1e12

    for iz, layer in enumerate(density):

        if not np.any(layer):
            continue

        # Depth of the cell center below the surface
        depth = (iz + 0.5) * dz_m

        # Receiver is above the surface
        separation = depth + observation_height_m

        kernel = gzz_kernel(
            separation,
            dx_m=dx_m,
            dy_m=dy_m
        )

        full = fftconvolve(
            layer,
            kernel,
            mode="full"
        )

        result += (
            full[
                63:63 + OBS_SHAPE[0],
                63:63 + OBS_SHAPE[1]
            ] * multiplier
        )

    return result


def calculate_gravity(
    density,
    component="gz",
    dx_m=CELL_SIZE,
    dy_m=CELL_SIZE,
    dz_m=CELL_SIZE,
    observation_height_m=0.0
):
    """Select the forward model using the specified physical geometry."""

    component = component.lower()

    kwargs = dict(
        dx_m=dx_m,
        dy_m=dy_m,
        dz_m=dz_m,
        observation_height_m=observation_height_m
    )

    if component == "gz":
        return calculate_gz(density, **kwargs)

    if component == "gzz":
        return calculate_gzz(density, **kwargs)

    raise ValueError("component must be 'gz' or 'gzz'")