"""Plot full-survey Thunderbird migration in one go.

Run from repository root:
    python -m src.visualization.plot_thunderbird_migration \
        --input src/data/Thunderbird_gravity_data/Tbird_gravity.dat \
        --output-root outputs/ml/field/Thunderbird_migration

Outputs:
    migration_volume.npy
    migration_topdown.png
    migration_3d.html
    gridded_gzz.npy
    x_obs_m.npy, y_obs_m.npy
    x_model_m.npy, y_model_m.npy, z_model_m.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import plotly.graph_objects as go

from src.data.e10_regional import (
    migration_gzz,
    OBS_SHAPE,
    SHAPE,
    DX,
    DY,
    DZ,
)


def load_gzz(path: Path, channel: int = 7):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("Expected 5-column Thunderbird data: x y z component value")

    rows = data[data[:, 3] == channel]
    if len(rows) == 0:
        raise ValueError(f"No rows found for channel {channel}")

    x = rows[:, 0]
    y = rows[:, 1]
    values = rows[:, 4]

    return x, y, values


def grid_observations(x, y, values):
    # Build a regular 161 x 161 grid across the full survey extent
    nx_obs = OBS_SHAPE[1]
    ny_obs = OBS_SHAPE[0]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    gx = np.linspace(xmin, xmax, nx_obs)
    gy = np.linspace(ymin, ymax, ny_obs)
    XX, YY = np.meshgrid(gx, gy)

    linear = LinearNDInterpolator(np.column_stack([x, y]), values, fill_value=np.nan)
    nearest = NearestNDInterpolator(np.column_stack([x, y]), values)

    gridded = linear(XX, YY)
    bad = ~np.isfinite(gridded)
    if np.any(bad):
        gridded[bad] = nearest(XX[bad], YY[bad])

    return gx, gy, np.asarray(gridded, dtype=np.float32)


def save_topdown_plot(migration, x_model, y_model, z_model, outpath: Path, slice_depth: float):
    # nearest depth layer
    k = int(np.argmin(np.abs(z_model - slice_depth)))
    depth = float(z_model[k])

    section = migration[k]
    scale = max(float(np.percentile(np.abs(section), 99.0)), 1e-8)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    im = ax.imshow(
        section,
        origin="lower",
        extent=[x_model[0] / 1000, x_model[-1] / 1000, y_model[0] / 1000, y_model[-1] / 1000],
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(f"Thunderbird migration top-down slice at depth = {depth:.1f} m")
    ax.set_xlabel("X from survey west (km)")
    ax.set_ylabel("Y from survey south (km)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Migration amplitude")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_plotly_3d(migration, x_model, y_model, z_model, outpath: Path, stride: int, percentile: float):
    # Downsample only for visualization
    mig = migration[:, ::stride, ::stride]
    xs = x_model[::stride] / 1000.0
    ys = y_model[::stride] / 1000.0
    zs = z_model

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")

    vals = mig.ravel()
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    vmax = max(float(np.percentile(np.abs(vals), percentile)), 1e-8)

    fig = go.Figure()

    # Positive migration
    pos = np.maximum(vals, 0.0)
    if np.max(pos) > 0:
        fig.add_trace(
            go.Volume(
                x=x,
                y=y,
                z=z,
                value=pos,
                isomin=0.20 * vmax,
                isomax=vmax,
                opacity=0.08,
                surface_count=10,
                colorscale="Reds",
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Positive migration",
            )
        )

    # Negative migration
    neg = np.maximum(-vals, 0.0)
    if np.max(neg) > 0:
        fig.add_trace(
            go.Volume(
                x=x,
                y=y,
                z=z,
                value=neg,
                isomin=0.20 * vmax,
                isomax=vmax,
                opacity=0.08,
                surface_count=10,
                colorscale="Blues",
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Negative migration",
            )
        )

    fig.update_layout(
        title="Thunderbird full-survey migration",
        width=950,
        height=800,
        scene=dict(
            xaxis_title="X from survey west (km)",
            yaxis_title="Y from survey south (km)",
            zaxis_title="Depth (m)",
            zaxis=dict(autorange="reversed"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.35),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=True,
    )

    fig.write_html(str(outpath), include_plotlyjs=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("outputs/ml/field/Thunderbird_migration"))
    p.add_argument("--channel", type=int, default=7, help="Thunderbird tensor channel for Gzz (default 7)")
    p.add_argument("--slice-depth", type=float, default=100.0, help="Depth in metres for top-down slice")
    p.add_argument("--stride", type=int, default=2, help="XY downsampling for Plotly only")
    p.add_argument("--percentile", type=float, default=99.0, help="Color/isosurface percentile for Plotly")
    args = p.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Thunderbird data...")
    x_obs_raw, y_obs_raw, gzz_raw = load_gzz(args.input, channel=args.channel)

    print("Gridding full survey...")
    x_obs, y_obs, gzz_grid = grid_observations(x_obs_raw, y_obs_raw, gzz_raw)

    if gzz_grid.shape != OBS_SHAPE:
        raise ValueError(f"Expected gridded Gzz shape {OBS_SHAPE}, got {gzz_grid.shape}")

    print("Computing full-survey migration in one go...")
    migration = migration_gzz(gzz_grid)

    if migration.shape != SHAPE:
        raise ValueError(f"Expected migration shape {SHAPE}, got {migration.shape}")

    # Model coordinates in metres, relative to survey west/south edges
    x_model = (np.arange(SHAPE[2]) + 0.5) * DX
    y_model = (np.arange(SHAPE[1]) + 0.5) * DY
    z_model = (np.arange(SHAPE[0]) + 0.5) * DZ

    print("Saving arrays...")
    np.save(out / "migration_volume.npy", migration)
    np.save(out / "gridded_gzz.npy", gzz_grid)
    np.save(out / "x_obs_m.npy", x_obs)
    np.save(out / "y_obs_m.npy", y_obs)
    np.save(out / "x_model_m.npy", x_model)
    np.save(out / "y_model_m.npy", y_model)
    np.save(out / "z_model_m.npy", z_model)

    print("Saving top-down plot...")
    save_topdown_plot(
        migration,
        x_model,
        y_model,
        z_model,
        out / "migration_topdown.png",
        slice_depth=args.slice_depth,
    )

    print("Saving Plotly 3D plot...")
    save_plotly_3d(
        migration,
        x_model,
        y_model,
        z_model,
        out / "migration_3d.html",
        stride=args.stride,
        percentile=args.percentile,
    )

    print("Done.")
    print("Saved:")
    print(" ", out / "migration_volume.npy")
    print(" ", out / "migration_topdown.png")
    print(" ", out / "migration_3d.html")


if __name__ == "__main__":
    main()