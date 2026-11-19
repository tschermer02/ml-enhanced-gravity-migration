"""E12 synthetic geology, isolated copy of E11 geometry rules."""
import numpy as np
from .config import SHAPE, MODEL_DX_M, MODEL_DZ_M
NZ,NY,NX=SHAPE
Z,Y,X=np.ogrid[:NZ,:NY,:NX]

def random_prism(rng, z_low=2, z_high=16):
    wx, wy = rng.integers(4, 17, size=2)
    wz = int(rng.integers(2, 9))
    x0 = int(rng.integers(2, NX - 2 - wx + 1))
    y0 = int(rng.integers(2, NY - 2 - wy + 1))
    z0 = int(rng.integers(z_low, min(z_high, NZ - wz) + 1))
    return x0, x0 + int(wx), y0, y0 + int(wy), z0, z0 + wz


def draw_prism(density, bounds, value):
    x0, x1, y0, y1, z0, z1 = bounds
    density[z0:z1, y0:y1, x0:x1] = value

def generate_broad_intrusion(rng, dx_m=MODEL_DX_M, dz_m=MODEL_DZ_M):
    """Generate one elongated, positive-density intrusive body."""

    for attempt in range(1000):

        length = float(rng.uniform(16, 40))
        width = float(rng.uniform(5, 14))
        thickness = float(rng.uniform(3, 10))

        strike = float(rng.uniform(0, np.pi))
        dip_deg = float(rng.uniform(-8, 8))
        dip = np.deg2rad(dip_deg)

        # Keep the rotated body inside the model.
        radius = np.hypot(length / 2, width / 2)
        margin = radius + 2

        xc = float(rng.uniform(margin, NX - margin))
        yc = float(rng.uniform(margin, NY - margin))

        top_center = float(rng.uniform(1, 10))

        # Cell-center coordinates.
        xx = np.arange(NX, dtype=np.float64) + 0.5
        yy = np.arange(NY, dtype=np.float64) + 0.5
        zz = np.arange(NZ, dtype=np.float64) + 0.5

        YY, XX = np.meshgrid(yy, xx, indexing="ij")

        # Rotate the horizontal coordinates.
        u = (
            (XX - xc) * np.cos(strike)
            + (YY - yc) * np.sin(strike)
        )

        v = (
            -(XX - xc) * np.sin(strike)
            + (YY - yc) * np.cos(strike)
        )

        # Elliptical horizontal footprint.
        footprint = (
            (u / (length / 2))**2
            + (v / (width / 2))**2
            <= 1.0
        )

        # Convert physical dip to vertical-cell displacement.
        vertical_slope = (
            np.tan(dip) * dx_m / dz_m
        )

        top = top_center + vertical_slope * u

        # Mild changes in thickness along the intrusion.
        thickness_field = thickness * (
            0.8 + 0.2 * np.cos(np.pi * u / length)
        )

        bottom = top + thickness_field

        # Reject bodies extending outside the model depth.
        if (
            np.min(top[footprint]) < 1.0
            or np.max(bottom[footprint]) > NZ - 1.0
        ):
            continue

        mask = (
            footprint[None, :, :]
            & (zz[:, None, None] >= top[None, :, :])
            & (zz[:, None, None] < bottom[None, :, :])
        )

        if np.count_nonzero(mask) < 50:
            continue

        density = np.zeros(SHAPE, dtype=np.float32)

        contrast = float(rng.uniform(0.2, 0.8))
        density[mask] = contrast

        return density

    raise RuntimeError(
        "Could not generate a valid broad intrusion"
    )


def separated(a, b, gap):
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return (ax1 + gap <= bx0 or bx1 + gap <= ax0 or
            ay1 + gap <= by0 or by1 + gap <= ay0 or
            az1 + gap <= bz0 or bz1 + gap <= az0)


def generate_density(geometry, rng, min_bodies=1, max_bodies=3, gap_cells=2,
                     density_sign="positive", dx_m=MODEL_DX_M, dz_m=MODEL_DZ_M):
    """Build a density volume using the original geometry recipes.

    `multiple` uses a configurable number of separated prisms. The remaining
    modes retain the shapes of the existing research generators.

    Salt domes are always negative and broad intrusions always positive.
    The remaining geometries follow the selected sign mode:
    positive, negative, or signed (random sign per sample).
    Basement geometry is not supported.
    """
    rho = np.zeros(SHAPE, dtype=np.float32)
    if geometry == "compact":
        draw_prism(rho, random_prism(rng), float(rng.uniform(.2, 1.)))
    elif geometry == "multiple":
        count = int(rng.integers(min_bodies, max_bodies + 1))
        bounds = []
        for n in range(count):
            for _ in range(1000):
                candidate = random_prism(rng)
                if all(separated(candidate, other, gap_cells) for other in bounds):
                    break
            else:
                raise RuntimeError(f"Could not place prism {n+1}; reduce body count or gap")
            draw_prism(rho, candidate, float(rng.uniform(.2, 1.)))
            bounds.append(candidate)
    elif geometry == "multiple_depths":
        for lo, hi in ((2, 7), (8, 14), (15, 22)):
            wx, wy = rng.integers(4, 16, size=2)
            wz = int(rng.integers(2, min(6, hi - lo) + 1))
            x0 = int(rng.integers(2, NX - 2 - wx + 1))
            y0 = int(rng.integers(2, NY - 2 - wy + 1))
            z0 = int(rng.integers(lo, hi - wz + 1))
            draw_prism(rho, (x0, x0 + int(wx), y0, y0 + int(wy), z0, z0 + wz),
                       float(rng.uniform(.2, 1.)))
    elif geometry == "dipping":
        x0 = int(rng.integers(6, 17))
        length = int(rng.integers(25, 43))
        y0 = int(rng.integers(8, 43))
        width = int(rng.integers(6, 15))
        z_start = float(rng.uniform(4., 8.))
        dip = float(rng.uniform(.16, .34)) * int(rng.choice([-1, 1]))
        if dip < 0:
            z_start = float(rng.uniform(14., 19.))
        thickness = float(rng.uniform(1., 2.2))
        mask = ((X >= x0) & (X < x0 + length) &
                (Y >= y0) & (Y < y0 + width) &
                (np.abs(Z - (z_start + dip * (X - x0))) <= thickness))
        rho = (mask * float(rng.uniform(.2, 1.))).astype(np.float32)
    elif geometry == "salt_dome":
        xc, yc = float(rng.uniform(22, 42)), float(rng.uniform(22, 42))
        r2 = (X - xc) ** 2 + (Y - yc) ** 2
        stem_r = float(rng.uniform(4., 7.))
        stem_top = int(rng.integers(6, 10))
        cap_top = int(rng.integers(2, 6))
        center = float(rng.uniform(7., 10.))
        cap_r = (float(rng.uniform(7., 10.)) + float(rng.uniform(2., 5.)) *
                 np.exp(-((Z - center) / float(rng.uniform(2., 3.8))) ** 2))
        mask = ((r2 <= stem_r ** 2) & (Z >= stem_top) & (Z < 23)) | \
               ((r2 <= cap_r ** 2) & (Z >= cap_top) & (Z <= 16))
        rho = (mask * float(rng.uniform(.2, .8))).astype(np.float32)
    elif geometry == "broad_intrusion":
        rho = generate_broad_intrusion(rng, dx_m=dx_m, dz_m=dz_m)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Salt domes are negative and broad intrusions positive irrespective of
    # --density. Other geometries follow the requested sign mode.
    if geometry == "salt_dome":
        rho = -np.abs(rho)
    elif geometry == "broad_intrusion":
        rho = np.abs(rho)
    elif density_sign == "negative":
        rho = -np.abs(rho)
    elif density_sign == "signed":
        if rng.integers(2) == 0:
            rho = -rho
    elif density_sign != "positive":
        raise ValueError(f"Unsupported density sign mode: {density_sign}")
    if rho.shape != SHAPE or not np.isfinite(rho).all() or not np.any(rho):
        raise ValueError(f"Invalid generated density: {geometry}")
    return rho


