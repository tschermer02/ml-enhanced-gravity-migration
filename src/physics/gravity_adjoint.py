"""Surface Gz adjoint integral and discrete transpose of the gravity forward model.

Coordinates: x east, y north, z positive downward, in meters.
Input Gz: (81, 81), indexed [y, x], in mGal.
Output: (24, 64, 64), indexed [z, y, x].

The paper's adjoint is a surface integral: gamma * integral_S f(r') Kz / R^3 dS.
Its discrete approximation includes the surface sample area dS.
The exact transpose of the cell-centered Python forward operator instead
includes the model cell volume and unit conversions. They are not identical.
"""

import numpy as np

# Keep these settings consistent with gravity_forward.py.
G = 6.67e-11                          # SI gravitational constant
NX, NY, NZ = 64, 64, 24
CELL_SIZE = 10.0                     # m
OBS_X = np.linspace(-85.0, 715.0, 81)
OBS_Y = np.linspace(-85.0, 715.0, 81)
OBS_Z = 0.0                          # m, positive downward
CELL_BATCH = 64                      # Control memory usage


def _backproject(gravity):
    """Compute sum_i Gz_i * (z_cell - z_obs) / distance_i**3 at each cell."""
    gravity = np.asarray(gravity, dtype=np.float64)
    if gravity.shape != (len(OBS_Y), len(OBS_X)):
        raise ValueError(f"Expected gravity shape {(len(OBS_Y), len(OBS_X))}")
    if not np.isfinite(gravity).all():
        raise ValueError("Gravity contains NaN or infinity")

    # Surface observation locations, flattened in [y, x] order.
    obs_x, obs_y = np.meshgrid(OBS_X, OBS_Y, indexing="xy")
    ox, oy = obs_x.ravel(), obs_y.ravel()
    data = gravity.ravel()

    # All model cell centers, flattened in [z, y, x] order.
    iz, iy, ix = np.indices((NZ, NY, NX))
    x = (ix.ravel() + 0.5) * CELL_SIZE
    y = (iy.ravel() + 0.5) * CELL_SIZE
    z = (iz.ravel() + 0.5) * CELL_SIZE

    output = np.empty(x.size, dtype=np.float64)

    # Batch over model cells to avoid a full (98304 x 6561) matrix.
    for start in range(0, x.size, CELL_BATCH):
        stop = min(start + CELL_BATCH, x.size)
        dx = x[start:stop, None] - ox[None, :]
        dy = y[start:stop, None] - oy[None, :]
        dz = z[start:stop, None] - OBS_Z
        r2 = dx**2 + dy**2 + dz**2
        if np.any(r2 == 0):
            raise ValueError("Receiver coincides with a model cell center")
        output[start:stop] = np.sum(data[None, :] * dz / (r2 * np.sqrt(r2)), axis=1)

    return output.reshape(NZ, NY, NX)


def calculate_adjoint(gravity):
    """Discretize A_z*(f)(r) = G * integral_S f(r') * Kz/R^3 dS.

    Here Kz = z_cell - z_observation, with +z downward.
    Uniform observation spacing gives dS = dx_obs * dy_obs.

    NOTE: input gravity is in mGal, so output is an *adjoint image*,
    not density in g/cm^3. The equation uses the SI value of G and no
    extra inverse/density conversion. Choose a consistent unit convention
    before interpreting its magnitude or using it in optimization.
    """
    dx_obs = float(OBS_X[1] - OBS_X[0])
    dy_obs = float(OBS_Y[1] - OBS_Y[0])
    return G * dx_obs * dy_obs * _backproject(gravity)


def calculate_discrete_transpose(gravity):
    """Exact A.T for gravity_forward.calculate_gravity(density).

    The forward maps density in g/cm^3 to gravity in mGal, using a
    point mass at each cell center. Its transpose includes the cell
    volume [m^3] and unit conversion factor 1e8.
    """
    return G * 1e8 * CELL_SIZE**3 * _backproject(gravity)


if __name__ == "__main__":
    # Demonstrate shapes without computing a forward-model dataset.
    example_data = np.zeros((len(OBS_Y), len(OBS_X)))
    example_data[len(OBS_Y)//2, len(OBS_X)//2] = 1.0
    adjoint = calculate_adjoint(example_data)
    print("Surface integral adjoint shape:", adjoint.shape)
    print("Maximum surface integral adjoint value:", adjoint.max())
