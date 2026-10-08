"""Plot surface gravity anomalies and slices through 3D density models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELL_M = 10.0
OBS_MIN_M = -85.0
OBS_MAX_M = 715.0


def plot_gravity(gravity: np.ndarray, output: str | Path) -> None:
    """Save a surface Gz anomaly map and a profile through its peak row."""
    gravity = np.asarray(gravity)
    if gravity.shape != (81, 81):
        raise ValueError(f"Expected gravity shape (81, 81), got {gravity.shape}")
    if not np.isfinite(gravity).all():
        raise ValueError("Gravity contains nonfinite values")

    x = np.linspace(OBS_MIN_M, OBS_MAX_M, gravity.shape[1])
    y = np.linspace(OBS_MIN_M, OBS_MAX_M, gravity.shape[0])
    peak_row, peak_col = np.unravel_index(np.argmax(np.abs(gravity)), gravity.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    im = axes[0].imshow(gravity, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                        cmap="viridis", aspect="equal")
    axes[0].plot(x[peak_col], y[peak_row], "rx", label="Peak |Gz|")
    axes[0].set(title="Surface gravity anomaly", xlabel="Easting (m)", ylabel="Northing (m)")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], label="Gz (mGal)")

    axes[1].plot(x, gravity[peak_row, :])
    axes[1].set(title=f"East–west profile at y = {y[peak_row]:g} m",
                xlabel="Easting (m)", ylabel="Gz (mGal)")
    axes[1].grid(alpha=0.3)
    _save(fig, output)


def plot_density_slices(density: np.ndarray, output: str | Path, *, cell_m: float = CELL_M) -> None:
    """Save horizontal, x-depth, and y-depth sections through the body center."""
    density = np.asarray(density)
    if density.ndim != 3 or not np.isfinite(density).all():
        raise ValueError("Density must be a finite (z, y, x) 3D array")
    nz, ny, nx = density.shape
    occupied = np.argwhere(density != 0)
    iz, iy, ix = np.rint(occupied.mean(axis=0)).astype(int) if occupied.size else (nz//2, ny//2, nx//2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    scale_max = max(float(np.max(np.abs(density))), 1e-12)
    panels = [
        (density[iz], [0, nx*cell_m, 0, ny*cell_m], "lower", f"Horizontal slice, z = {(iz+0.5)*cell_m:g} m", "Easting (m)", "Northing (m)"),
        (density[:, iy, :], [0, nx*cell_m, nz*cell_m, 0], "upper", f"X–depth slice, y = {(iy+0.5)*cell_m:g} m", "Easting (m)", "Depth (m)"),
        (density[:, :, ix], [0, ny*cell_m, nz*cell_m, 0], "upper", f"Y–depth slice, x = {(ix+0.5)*cell_m:g} m", "Northing (m)", "Depth (m)"),
    ]
    for ax, (values, extent, origin, title, xlabel, ylabel) in zip(axes, panels):
        im = ax.imshow(values, origin=origin, extent=extent, cmap="viridis", vmin=0, vmax=scale_max, aspect="auto")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Density contrast (g/cm³)")
    _save(fig, output)


def plot_volume_comparison(truth: np.ndarray, estimate: np.ndarray,
                           output: str | Path, *, cell_m: float = CELL_M) -> None:
    """Compare central x-depth sections of true and estimated 3D volumes.

    Note: if estimate is an adjoint image, its values are NOT density units.
    Each panel therefore has a separate scale.
    """
    truth, estimate = np.asarray(truth), np.asarray(estimate)
    if truth.shape != estimate.shape or truth.ndim != 3:
        raise ValueError("Truth and estimate must be same-shaped 3D volumes")
    nz, ny, nx = truth.shape
    occupied = np.argwhere(truth != 0)
    iy = int(np.rint(occupied[:, 1].mean())) if occupied.size else ny // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, vol, label in zip(axes, [truth, estimate], ["True density", "Estimated / adjoint image"]):
        im = ax.imshow(vol[:, iy, :], origin="upper", extent=[0, nx*cell_m, nz*cell_m, 0], aspect="auto")
        ax.set(title=label, xlabel="Easting (m)", ylabel="Depth (m)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="g/cm³" if label == "True density" else "Adjoint / estimate value")
    _save(fig, output)


def _save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sample_plots(sample_name: str, gravity: np.ndarray, density: np.ndarray,
                      *, root: str | Path = "results/figures") -> None:
    """Save both gravity and density plots inside a sample-named subfolder.

    Example: results/figures/sample_000000/gravity.png and density.png
    """
    sample_name = str(sample_name)
    folder = Path(root) / sample_name
    plot_gravity(gravity, folder / "gravity.png")
    plot_density_slices(density, folder / "density.png")
