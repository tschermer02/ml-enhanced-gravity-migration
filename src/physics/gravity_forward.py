
"""
Calculate surface vertical gravity (Gz) from a 3D density model.

Uses a cell-centered point-mass approximation based on the
translated MATLAB gravity forward model.

Density: (24, 64, 64) [z, y, x] in g/cm^3
Gravity: (81, 81) [y, x] in mGal

Coordinates: x east, y north, z positive downward.
"""

import numpy as np


# ============================================================
# 1. PARAMETERS
# ============================================================

# Gravitational constant [m^3/(kg s^2)]
G = 6.67e-11

# Grid dimensions
NX = 64
NY = 64
NZ = 24
CELL_SIZE = 10.0

# Observation coordinates [m]
OBS_X = np.linspace(-85, 715, 81)
OBS_Y = np.linspace(-85, 715, 81)
OBS_Z = 0.0

# Number of observation points processed per batch
BATCH_SIZE = 128


# ============================================================
# 2. GRAVITY FORWARD MODEL
# ============================================================

def calculate_gravity(density):
    """Calculate surface Gz from a 3D density model."""

    # Check density model dimensions
    if density.shape != (NZ, NY, NX):
        raise ValueError("Incorrect density model dimensions")

    # Find cells with nonzero density contrast
    iz, iy, ix = np.nonzero(density)

    # Return zero gravity if the density model is empty
    if len(iz) == 0:
        return np.zeros((len(OBS_Y), len(OBS_X)))

    # Cell center coordinates [m]
    x = (ix + 0.5) * CELL_SIZE
    y = (iy + 0.5) * CELL_SIZE
    z = (iz + 0.5) * CELL_SIZE

    # Convert density contrast into gravitational mass factor
    # g/cm^3 -> kg/m^3: multiply by 1000
    # m/s^2 -> mGal: multiply by 1e5
    mass = G * density[iz, iy, ix] * CELL_SIZE**3 * 1e8

    # Create surface observation grid
    xx, yy = np.meshgrid(OBS_X, OBS_Y)

    gx = xx.ravel()
    gy = yy.ravel()

    # Initialize gravity output
    gz = np.zeros(gx.size)

    # Calculate gravity at each observation point
    for start in range(0, gx.size, BATCH_SIZE):

        end = min(start + BATCH_SIZE, gx.size)

        # Distance from observation points to density cells
        dx = x[None, :] - gx[start:end, None]
        dy = y[None, :] - gy[start:end, None]
        dz = z[None, :] - OBS_Z

        # Distance squared
        r2 = dx**2 + dy**2 + dz**2

        # Vertical gravity contribution
        gz[start:end] = np.sum(
            mass[None, :] * dz / (r2 * np.sqrt(r2)),
            axis=1
        )

    # Return gravity as an 81 x 81 array
    return gz.reshape(len(OBS_Y), len(OBS_X))


# ============================================================
# 3. TEST FORWARD MODEL
# ============================================================

if __name__ == "__main__":

    # Create a simple test density model
    density = np.zeros((NZ, NY, NX))
    density[2:5, 30:36, 30:36] = 0.5

    # Calculate surface gravity
    gravity = calculate_gravity(density)

    print("Density shape:", density.shape)
    print("Gravity shape:", gravity.shape)
    print("Maximum Gz:", gravity.max(), "mGal")
