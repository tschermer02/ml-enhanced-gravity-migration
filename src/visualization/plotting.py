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
                           output: str | Path, *, cell_m: float = CELL_M,
                           estimate_label: str = "Estimated / adjoint image",
                           estimate_units: str = "Adjoint / estimate value",
                           title: str | None = None) -> None:
    """Plot matching orthogonal slices of the true and estimated 3D volumes."""
    truth, estimate = np.asarray(truth), np.asarray(estimate)
    if truth.shape != estimate.shape or truth.ndim != 3:
        raise ValueError("Truth and estimate must be same-shaped 3D volumes")
    nz, ny, nx = truth.shape
    occupied = np.argwhere(truth != 0)
    iz, iy, ix = (
        np.rint(occupied.mean(axis=0)).astype(int)
        if occupied.size else (nz // 2, ny // 2, nx // 2)
    )
    slices = [
        (truth[iz], estimate[iz], [0, nx*cell_m, 0, ny*cell_m], "lower",
         f"Horizontal slice, z = {(iz + 0.5)*cell_m:g} m", "Easting (m)", "Northing (m)"),
        (truth[:, iy, :], estimate[:, iy, :], [0, nx*cell_m, nz*cell_m, 0], "upper",
         f"X-depth slice, y = {(iy + 0.5)*cell_m:g} m", "Easting (m)", "Depth (m)"),
        (truth[:, :, ix], estimate[:, :, ix], [0, ny*cell_m, nz*cell_m, 0], "upper",
         f"Y-depth slice, x = {(ix + 0.5)*cell_m:g} m", "Northing (m)", "Depth (m)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    row_data = ((truth, "Synthetic density", "Density contrast (g/cm³)"),
                (estimate, estimate_label, estimate_units))
    for row, (volume, row_label, units) in enumerate(row_data):
        scale_min = min(float(volume.min()), 0.0)
        scale_max = max(float(volume.max()), 1e-12)
        last_image = None
        for column, (true_slice, estimate_slice, extent, origin, slice_title,
                     xlabel, ylabel) in enumerate(slices):
            values = true_slice if row == 0 else estimate_slice
            last_image = axes[row, column].imshow(
                values, origin=origin, extent=extent, cmap="viridis",
                vmin=scale_min, vmax=scale_max, aspect="auto",
            )
            axes[row, column].set(
                title=slice_title, xlabel=xlabel, ylabel=ylabel,
            )
        fig.colorbar(last_image, ax=axes[row, :], shrink=0.85, label=units)
        axes[row, 0].annotate(
            row_label, xy=(-0.2, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=12,
        )
    if title:
        fig.suptitle(title)
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

# Cross-section comparison tools (from export_three_panel_cross_sections.py)
from scipy import ndimage
from matplotlib.colors import TwoSlopeNorm
SHAPE = (24, 64, 64)
DENSITY_THRESHOLD = 0.1
COLUMNS = ('Synthetic Model', 'Migration', 'Inversion')
ROWS = ('XY', 'XZ', 'YZ')

def _name(value: str) -> str:
    value = str(value).removesuffix('.npz')
    if value.startswith('sample_'):
        return value
    if value.isdigit():
        return f'sample_{int(value):06d}'
    return value


def _load(path: Path, migration_source: str):
    with np.load(path) as archive:
        required = ('density', 'prediction')
        for key in required:
            if key not in archive:
                raise KeyError(f'{path}: missing {key}; available keys: {archive.files}')
        density = np.asarray(archive['density'], dtype=np.float32).squeeze()
        inversion = np.asarray(archive['prediction'], dtype=np.float32).squeeze()
        if migration_source == 'normalized':
            if 'migration_input' in archive:
                migration = np.asarray(archive['migration_input'], dtype=np.float32).squeeze()
            elif 'migration' in archive:
                raw = np.asarray(archive['migration'], dtype=np.float32).squeeze()
                migration = raw / max(float(np.max(np.abs(raw))), 1e-8)
            else:
                raise KeyError(f'{path}: no migration_input or migration field')
        else:
            if 'migration' not in archive:
                raise KeyError(f'{path}: no raw migration field')
            migration = np.asarray(archive['migration'], dtype=np.float32).squeeze()
    for key, arr in [('density', density), ('migration', migration), ('prediction', inversion)]:
        if arr.shape != SHAPE or not np.isfinite(arr).all():
            raise ValueError(f'{path}: {key} has invalid shape/values {arr.shape}')
    return density, migration, inversion


def _components(density, threshold, connectivity, min_body_voxels):
    mask = np.abs(density) >= threshold
    struct = ndimage.generate_binary_structure(3, connectivity)
    labeled, n = ndimage.label(mask, structure=struct)
    bodies = []
    for identifier in range(1, n+1):
        indices = np.argwhere(labeled == identifier)
        if len(indices) < min_body_voxels:
            continue
        centroid = np.mean(indices, axis=0)
        center = tuple(int(np.clip(np.round(v), 0, d-1)) for v,d in zip(centroid, SHAPE))
        bodies.append((len(indices), center, identifier))
    bodies.sort(key=lambda item: (-item[0], item[1]))
    return bodies


def _slices(volume, center):
    z, y, x = center
    return [volume[z, :, :], volume[:, y, :], volume[:, :, x]]


def _figure(truth, mig, pred, center, title, output, migration_source, dpi):
    volumes = [truth, mig, pred]
    nz, ny, nx = truth.shape
    density_max = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 0.1)
    density_norm = TwoSlopeNorm(vmin=-density_max, vcenter=0, vmax=density_max)
    migmax = max(float(np.max(np.abs(mig))), 1e-6)
    mig_norm = TwoSlopeNorm(vmin=-migmax, vcenter=0, vmax=migmax)
    fig, axes = plt.subplots(3, 3, figsize=(14.4, 11.2), constrained_layout=True,
                             gridspec_kw={'height_ratios':[1.0, 0.78, 0.78]})
    for col, volume in enumerate(volumes):
        for row, section in enumerate(_slices(volume, center)):
            ax = axes[row, col]
            # z=0 at top for vertical slices, x/y show geographic increasing coordinates.
            if row == 0:
                extent = [0, nx*CELL_M, 0, ny*CELL_M]
                origin = 'lower'
                xlabel, ylabel = 'East (m)', 'North (m)'
            elif row == 1:
                extent = [0, nx*CELL_M, nz*CELL_M, 0]
                origin = 'upper'
                xlabel, ylabel = 'East (m)', 'Depth (m)'
            else:
                extent = [0, ny*CELL_M, nz*CELL_M, 0]
                origin = 'upper'
                xlabel, ylabel = 'North (m)', 'Depth (m)'
            im = ax.imshow(section, cmap='RdBu_r',
                           norm=mig_norm if col == 1 else density_norm,
                           origin=origin, extent=extent, interpolation='nearest',
                           aspect='auto')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            if col == 0:
                z,y,x=center
                ax.text(0.02, 0.97, [f'XY · depth {(z+0.5)*CELL_M:.0f} m',
                                      f'XZ · north {(y+0.5)*CELL_M:.0f} m',
                                      f'YZ · east {(x+0.5)*CELL_M:.0f} m'][row],
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(facecolor='white',alpha=.82,edgecolor='none',pad=3))
            if row == 0: ax.set_title(COLUMNS[col], fontsize=14, weight='bold', pad=12)
    bar_truth = fig.colorbar(axes[0, 0].images[0], ax=[axes[r,c] for c in (0,2) for r in range(3)],
                             orientation='horizontal', shrink=.72, pad=.08, aspect=38)
    bar_truth.set_label('Density contrast (g/cm³) · shared true/pred scale', fontsize=10)
    bar_mig = fig.colorbar(axes[0, 1].images[0], ax=[axes[r,1] for r in range(3)],
                           orientation='horizontal', shrink=.83, pad=.08, aspect=18)
    bar_mig.set_label('Normalized migration (unitless)' if migration_source == 'normalized'
                      else 'Migration amplitude (not calibrated density)', fontsize=10)
    fig.suptitle(title, fontsize=16, weight='bold')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output


def export_one(path, prediction_root, split, threshold, connectivity,
               min_body_voxels, migration_source, dpi, overwrite, per_body):
    truth, migration, prediction = _load(path, migration_source)
    bodies = _components(truth, threshold, connectivity, min_body_voxels)
    if not bodies:
        raise ValueError(f'{path}: no true bodies at threshold {threshold}')
    relative = path.relative_to(prediction_root)
    # predictions/test/<sample>.npz -> figures/test/<sample>_cross_sections.png
    parts = relative.parts
    if len(parts) < 3 or parts[-3:-1] != ('predictions', split):
        raise ValueError(f'Unexpected prediction layout: {path}')
    experiment_parts = parts[:-3]
    outdir = prediction_root.joinpath(*experiment_parts, 'figures', split)
    center = tuple(int(np.round(v)) for v in np.mean(np.argwhere(np.abs(truth)>=threshold),axis=0))
    outputs = []
    overview = outdir / f'{path.stem}_cross_sections.png'
    if overwrite or not overview.exists():
        outputs.append(_figure(truth,migration,prediction,center,
                    f'{path.stem} | Full-model cross-sections', overview,migration_source,dpi))
    if per_body:
        for number, (count, body_center, _) in enumerate(bodies, 1):
            output = outdir / f'{path.stem}_body_{number:02d}_cross_sections.png'
            if overwrite or not output.exists():
                outputs.append(_figure(truth,migration,prediction,body_center,
                    f'{path.stem} | Body {number} ({count} voxels) — matched cross-sections',
                    output,migration_source,dpi))
    return outputs, len(bodies)

