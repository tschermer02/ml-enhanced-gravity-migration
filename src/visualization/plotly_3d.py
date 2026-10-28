"""Voxel-accurate 3D true density and signed CNN prediction visualizations.

Works with positive-only E01-E06 and signed E06_signed density volumes.
Density axes are (z, y, x), units g/cm^3; depth is positive downward.
A .gif path with make_gif=True saves GIF only. Requires Plotly, Kaleido,
Pillow, and compatible Chrome/Chromium for GIF rendering.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CELL_M = 10.0
# True cells with |density| <= this are treated as background.
TRUE_DENSITY_THRESHOLD = 0.1


def _domain_edges(nx: int, ny: int, nz: int):
    x1, y1, z1 = nx * CELL_M, ny * CELL_M, nz * CELL_M
    corners = [(0, 0, 0), (x1, 0, 0), (x1, y1, 0), (0, y1, 0),
               (0, 0, z1), (x1, 0, z1), (x1, y1, z1), (0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xx, yy, zz = [], [], []
    for a, b in edges:
        for v, dest in zip(corners[a], (xx, yy, zz)):
            dest.append(v)
        for v, dest in zip(corners[b], (xx, yy, zz)):
            dest.append(v)
        xx.append(None)
        yy.append(None)
        zz.append(None)
    return go.Scatter3d(x=xx, y=yy, z=zz, mode="lines",
                        line=dict(color="gray", width=3),
                        showlegend=False, hoverinfo="skip")


def _voxel_surface(mask: np.ndarray, *, color: str, name: str):
    """Triangulate exposed voxel faces for one signed occupancy class.

    Quads are generated in vectorized batches; no rectangular bounding boxes.
    Signs get independent meshes so adjacent positive/negative bodies are visible.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Occupancy mask must be (z,y,x)")
    if not mask.any():
        return None
    nz, ny, nx = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    faces = [
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ]
    vertices = []
    triangles = []
    vertex_offset = 0
    for (dx, dy, dz), corners in faces:
        neighbor = padded[1 + dz:1 + dz + nz,
                          1 + dy:1 + dy + ny,
                          1 + dx:1 + dx + nx]
        voxel_zyx = np.argwhere(mask & ~neighbor)
        count = len(voxel_zyx)
        if not count:
            continue
        base_xyz = voxel_zyx[:, [2, 1, 0]].astype(np.float32)
        corner_offsets = np.asarray(corners, dtype=np.float32)
        vertices.append(((base_xyz[:, None, :] + corner_offsets) * CELL_M)
                        .reshape(-1, 3))
        start = vertex_offset + 4 * np.arange(count, dtype=np.int32)
        triangles.append(np.stack((start, start + 1, start + 2), axis=1))
        triangles.append(np.stack((start, start + 2, start + 3), axis=1))
        vertex_offset += 4 * count
    verts = np.concatenate(vertices, axis=0)
    tris = np.concatenate(triangles, axis=0)
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                     color=color, opacity=0.85, flatshading=True,
                     lighting=dict(ambient=0.55, diffuse=0.75, specular=0.08),
                     name=name, showlegend=True, hoverinfo="name")


def _true_density_surfaces(density: np.ndarray):
    """One accurate voxel surface per sign, supporting signed salt domes."""
    positive = _voxel_surface(density >= TRUE_DENSITY_THRESHOLD,
                              color="royalblue", name="True positive contrast")
    negative = _voxel_surface(density <= -TRUE_DENSITY_THRESHOLD,
                              color="deepskyblue", name="True negative contrast")
    return [trace for trace in (positive, negative) if trace is not None]


def _prediction_surfaces(reconstruction: np.ndarray, threshold: float):
    """Render positive and negative predictions independently.

    The supplied threshold is relative to the peak amplitude OF EACH SIGN.
    Negative predictions use -density as their magnitude, so a signed model's
    negative salt dome does not disappear when max(density) is near zero.
    """
    z, y, x = np.indices(reconstruction.shape)
    coords = dict(x=((x + 0.5) * CELL_M).ravel(),
                  y=((y + 0.5) * CELL_M).ravel(),
                  z=((z + 0.5) * CELL_M).ravel())
    outputs = []
    sign_specs = [
        (np.maximum(reconstruction, 0), "Predicted positive contrast", "YlOrRd"),
        (np.maximum(-reconstruction, 0), "Predicted negative contrast", "Blues"),
    ]
    for amplitudes, name, colors in sign_specs:
        peak = float(np.max(amplitudes))
        # Prevent tiny signed numerical noise from producing a visual body.
        if peak <= 1e-5:
            continue
        outputs.append(go.Isosurface(
            **coords, value=(amplitudes / peak).ravel(),
            isomin=threshold, isomax=1.0, surface_count=3,
            colorscale=colors, opacity=0.55, showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=name, showlegend=True,
            hovertemplate="Relative |density|=%{value:.3f}<extra>" + name + "</extra>",
        ))
    return outputs


def plot_sample_3d(
    density, gravity, output_path, reconstruction=None,
    reconstruction_threshold=0.25, make_gif=True, gif_frames=80,
    gif_duration_seconds=8.0,
):
    """Render signed ground truth and optional signed CNN prediction.

    gravity is retained for backwards compatibility but is not plotted.
    reconstruction_threshold is a fraction of each sign's predicted peak.
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

    titles = ["True density contrast"]
    specs = [[{"type": "scene"}]]
    if reconstruction is not None:
        titles.append("CNN prediction (signed relative amplitude)")
        specs[0].append({"type": "scene"})
    fig = make_subplots(rows=1, cols=len(titles), specs=specs,
                        subplot_titles=titles, horizontal_spacing=0.06)
    fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=1)
    for trace in _true_density_surfaces(density):
        fig.add_trace(trace, row=1, col=1)

    if reconstruction is not None:
        fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=2)
        for trace in _prediction_surfaces(reconstruction, reconstruction_threshold):
            fig.add_trace(trace, row=1, col=2)

    scene_style = dict(
        xaxis=dict(title="East (m)", range=[0, nx * CELL_M]),
        yaxis=dict(title="North (m)", range=[0, ny * CELL_M]),
        zaxis=dict(title="Depth (m)", range=[nz * CELL_M, 0]),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.55),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    fig.update_layout(scene=scene_style, title="Synthetic density contrast model",
                      width=1300 if reconstruction is not None else 850,
                      height=650, margin=dict(l=15, r=30, t=80, b=25))
    if reconstruction is not None:
        fig.update_layout(scene2=scene_style)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if make_gif and output_path.suffix.lower() == ".gif":
        save_rotation_gif(fig, output_path, frames=gif_frames,
                          duration_seconds=gif_duration_seconds,
                          has_reconstruction=reconstruction is not None)
        return output_path
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    if make_gif:
        save_rotation_gif(fig, output_path.with_suffix(".gif"), frames=gif_frames,
                          duration_seconds=gif_duration_seconds,
                          has_reconstruction=reconstruction is not None)
    return output_path


def save_rotation_gif(fig, output_path, frames=80, duration_seconds=8.0,
                      has_reconstruction=False):
    """Render rotating Plotly frames using Kaleido and Pillow."""
    from PIL import Image

    if frames < 2 or duration_seconds <= 0:
        raise ValueError("frames must be >= 2 and duration_seconds must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in range(frames):
        angle = 2 * math.pi * frame / frames
        camera = dict(eye=dict(x=2.05 * math.cos(angle),
                               y=2.05 * math.sin(angle), z=1.0))
        fig.update_layout(scene_camera=camera)
        if has_reconstruction:
            fig.update_layout(scene2_camera=camera)
        try:
            png = fig.to_image(format="png", width=1600, height=900, scale=1)
        except Exception as exc:
            raise RuntimeError(
                "Plotly GIF rendering failed. Check Kaleido and Chrome/Chromium "
                "compatibility. Original error: " + str(exc)
            ) from exc
        with Image.open(BytesIO(png)) as frame_image:
            images.append(frame_image.convert("RGB"))
        print(f"Rendered frame {frame + 1}/{frames}", flush=True)
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=round(duration_seconds * 1000 / frames),
                   loop=0, disposal=2, optimize=False)
    return output_path


def save_sample_3d(sample_name: str, density: np.ndarray, gravity: np.ndarray,
                   *, root: str | Path = "results/figures",
                   reconstruction: np.ndarray | None = None,
                   make_gif: bool = True, gif_frames: int = 80,
                   gif_duration_seconds: float = 8.0) -> Path:
    """Save 3D view under a sample-named subdirectory."""
    sample_dir = Path(root) / str(sample_name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / ("interactive_3d.gif" if make_gif else "interactive_3d.html")
    return plot_sample_3d(density, gravity, path,
                          reconstruction=reconstruction, make_gif=make_gif,
                          gif_frames=gif_frames,
                          gif_duration_seconds=gif_duration_seconds)

# Matplotlib prism overview (from visualization.py)
import matplotlib.pyplot as plt
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

