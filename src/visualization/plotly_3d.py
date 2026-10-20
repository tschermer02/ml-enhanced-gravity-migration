"""Plotly views and GIF-only exports for single- and multi-body 3D density models.

Density volume shape: (z, y, x). Coordinates are in meters and depth is positive downward.
For GIF-only output, pass a path ending in .gif with make_gif=True.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import label

CELL_M = 10.0


def _domain_edges(nx: int, ny: int, nz: int):
    """Wireframe of the full 3D model domain."""
    x1, y1, z1 = nx * CELL_M, ny * CELL_M, nz * CELL_M
    corners = [
        (0, 0, 0), (x1, 0, 0), (x1, y1, 0), (0, y1, 0),
        (0, 0, z1), (x1, 0, z1), (x1, y1, z1), (0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
        (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xx, yy, zz = [], [], []
    for a, b in edges:
        for value, dest in zip(corners[a], (xx, yy, zz)):
            dest.append(value)
        for value, dest in zip(corners[b], (xx, yy, zz)):
            dest.append(value)
        xx.append(None)
        yy.append(None)
        zz.append(None)
    return go.Scatter3d(
        x=xx, y=yy, z=zz, mode="lines",
        line=dict(color="gray", width=3),
        showlegend=False, hoverinfo="skip",
    )


def _prism_meshes(density: np.ndarray):
    """Draw one true prism per 3D connected component, not a combined bounding box.

    Uses 6-neighbor connectivity to avoid merging diagonally touching bodies.
    Expected synthetic bodies are solid axis-aligned prisms separated by gaps.
    """
    mask = np.asarray(density) > 0
    # 6-connected voxel components: face-sharing cells belong to one body.
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    labeled, body_count = label(mask, structure=structure)

    meshes = []
    for body_id in range(1, body_count + 1):
        iz, iy, ix = np.where(labeled == body_id)
        if ix.size == 0:
            continue
        x0, x1 = float(ix.min() * CELL_M), float((ix.max() + 1) * CELL_M)
        y0, y1 = float(iy.min() * CELL_M), float((iy.max() + 1) * CELL_M)
        z0, z1 = float(iz.min() * CELL_M), float((iz.max() + 1) * CELL_M)
        x = [x0, x1, x1, x0, x0, x1, x1, x0]
        y = [y0, y0, y1, y1, y0, y0, y1, y1]
        z = [z0, z0, z0, z0, z1, z1, z1, z1]
        meshes.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
            j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
            k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
            color="royalblue", opacity=0.85, flatshading=True,
            name=f"True body {body_id}", showlegend=False,
            hovertemplate=(
                "x=%{x:.0f} m<br>y=%{y:.0f} m<br>depth=%{z:.0f} m"
                f"<extra>True body {body_id}</extra>"
            ),
        ))
    return meshes


def plot_sample_3d(
    density,
    gravity,
    output_path,
    reconstruction=None,
    reconstruction_threshold=0.25,
    make_gif=True,
    gif_frames=80,
    gif_duration_seconds=8.0,
):
    """Plot true density bodies and an optional CNN reconstruction.

    ``gravity`` is retained for compatibility with existing callers but unused.
    ``reconstruction_threshold`` is a fraction of maximum positive prediction.
    When ``output_path`` has suffix .gif and make_gif=True, NO HTML is written.
    """
    density = np.asarray(density)
    if density.ndim != 3 or not np.isfinite(density).all():
        raise ValueError("True density must be a finite 3D array")
    nz, ny, nx = density.shape
    if reconstruction is not None:
        reconstruction = np.asarray(reconstruction)
        if reconstruction.shape != density.shape or not np.isfinite(reconstruction).all():
            raise ValueError("Reconstruction must be finite and match density shape")
        if not 0 < reconstruction_threshold < 1:
            raise ValueError("reconstruction_threshold must be between 0 and 1")

    titles = ["True density bodies"]
    specs = [[{"type": "scene"}]]
    if reconstruction is not None:
        titles.append("CNN prediction (relative amplitude)")
        specs[0].append({"type": "scene"})
    fig = make_subplots(
        rows=1, cols=len(titles), specs=specs, subplot_titles=titles,
        horizontal_spacing=0.06,
    )
    fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=1)
    for prism in _prism_meshes(density):
        fig.add_trace(prism, row=1, col=1)

    if reconstruction is not None:
        fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=2)
        peak = float(reconstruction.max())
        if peak > 0:
            z, y, x = np.indices(reconstruction.shape)
            fig.add_trace(go.Isosurface(
                x=((x + 0.5) * CELL_M).ravel(),
                y=((y + 0.5) * CELL_M).ravel(),
                z=((z + 0.5) * CELL_M).ravel(),
                value=(reconstruction / peak).ravel(),
                isomin=reconstruction_threshold, isomax=1,
                surface_count=3, colorscale="Plasma", opacity=0.55,
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                hovertemplate="Relative amplitude=%{value:.3f}<extra></extra>",
            ), row=1, col=2)

    scene_style = dict(
        xaxis=dict(title="East (m)", range=[0, nx * CELL_M]),
        yaxis=dict(title="North (m)", range=[0, ny * CELL_M]),
        zaxis=dict(title="Depth (m)", range=[nz * CELL_M, 0]),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.55),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    fig.update_layout(
        scene=scene_style, title="Synthetic density model",
        width=1300 if reconstruction is not None else 850,
        height=650, margin=dict(l=15, r=30, t=80, b=25),
    )
    if reconstruction is not None:
        fig.update_layout(scene2=scene_style)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if make_gif and output_path.suffix.lower() == ".gif":
        save_rotation_gif(fig, output_path, frames=gif_frames,
                          duration_seconds=gif_duration_seconds,
                          has_reconstruction=reconstruction is not None)
        return output_path

    # Preserve the original behavior for callers requesting HTML.
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    if make_gif:
        save_rotation_gif(fig, output_path.with_suffix(".gif"), frames=gif_frames,
                          duration_seconds=gif_duration_seconds,
                          has_reconstruction=reconstruction is not None)
    return output_path


def save_rotation_gif(fig, output_path, frames=80, duration_seconds=8.0,
                      has_reconstruction=False):
    """Render a rotating 3D GIF via Plotly/Kaleido and Pillow."""
    from PIL import Image

    if frames < 2 or duration_seconds <= 0:
        raise ValueError("frames must be >= 2 and duration_seconds must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in range(frames):
        angle = 2 * math.pi * frame / frames
        camera = dict(eye=dict(
            x=2.05 * math.cos(angle),
            y=2.05 * math.sin(angle), z=1.0,
        ))
        fig.update_layout(scene_camera=camera)
        if has_reconstruction:
            fig.update_layout(scene2_camera=camera)
        try:
            png = fig.to_image(format="png", width=1600, height=900, scale=1)
        except Exception as exc:
            raise RuntimeError(
                "Plotly GIF rendering failed. Check Kaleido and Chrome/Chromium "
                "compatibility on this machine. Original error: " + str(exc)
            ) from exc
        with Image.open(BytesIO(png)) as frame_image:
            images.append(frame_image.convert("RGB"))
        print(f"Rendered frame {frame + 1}/{frames}", flush=True)
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=round(duration_seconds * 1000 / frames),
        loop=0, disposal=2, optimize=False,
    )
    return output_path


def save_sample_3d(sample_name: str, density: np.ndarray, gravity: np.ndarray,
                   *, root: str | Path = "results/figures",
                   reconstruction: np.ndarray | None = None,
                   make_gif: bool = True, gif_frames: int = 80,
                   gif_duration_seconds: float = 8.0) -> Path:
    """Save 3D visualization under a sample-named subdirectory."""
    sample_dir = Path(root) / str(sample_name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / ("interactive_3d.gif" if make_gif else "interactive_3d.html")
    return plot_sample_3d(
        density, gravity, path, reconstruction=reconstruction,
        make_gif=make_gif, gif_frames=gif_frames,
        gif_duration_seconds=gif_duration_seconds,
    )
