"""3D visualization of a single rectangular density prism."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _faces(x0, x1, y0, y1, z0, z1):
    v = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    return [[v[i] for i in face] for face in
            [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]


def plot_prism_3d(density: np.ndarray, output: str | Path, *, cell_m: float = 10.0):
    """Show the bounds of a single nonzero prism in the full model domain."""
    density = np.asarray(density)
    if density.ndim != 3 or not np.isfinite(density).all():
        raise ValueError("Expected a finite (z, y, x) density array")
    occupied = np.argwhere(density != 0)
    if occupied.size == 0:
        raise ValueError("Density model contains no prism")
    lo = occupied.min(axis=0) * cell_m
    hi = (occupied.max(axis=0) + 1) * cell_m
    nz, ny, nx = density.shape

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    body = Poly3DCollection(_faces(lo[2], hi[2], lo[1], hi[1], lo[0], hi[0]),
                            facecolor="tab:orange", edgecolor="black", alpha=0.8)
    ax.add_collection3d(body)
    ax.set(xlim=(0,nx*cell_m), ylim=(0,ny*cell_m), zlim=(nz*cell_m,0),
           xlabel="Easting (m)", ylabel="Northing (m)", zlabel="Depth (m)",
           title="Synthetic density prism")
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=25, azim=-55)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
