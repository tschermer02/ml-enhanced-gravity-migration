"""Interactive Plotly views of synthetic gravity data and buried density bodies.

Optional reconstructions are displayed using relative amplitude, NOT as calibrated
physical density unless the reconstruction already has those units.
"""

from pathlib import Path
from io import BytesIO
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CELL_M = 10.0


def _domain_edges(nx, ny, nz):
    """Return a wireframe representing the full model domain."""
    x1, y1, z1 = nx * CELL_M, ny * CELL_M, nz * CELL_M
    corners = [(0, 0, 0), (x1, 0, 0), (x1, y1, 0), (0, y1, 0),
               (0, 0, z1), (x1, 0, z1), (x1, y1, z1), (0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xx, yy, zz = [], [], []
    for a, b in edges:
        for value, dest in zip(corners[a], (xx, yy, zz)):
            dest.append(value)
        for value, dest in zip(corners[b], (xx, yy, zz)):
            dest.append(value)
        xx.append(None); yy.append(None); zz.append(None)
    return go.Scatter3d(x=xx, y=yy, z=zz, mode="lines",
                        line=dict(color="gray", width=3),
                        showlegend=False, hoverinfo="skip")


def _prism_mesh(density):
    """Draw the bounding prism of the nonzero cells (single-prism datasets)."""
    iz, iy, ix = np.nonzero(density)
    if not len(ix):
        raise ValueError("Density model contains no nonzero body")
    x0, x1 = ix.min() * CELL_M, (ix.max() + 1) * CELL_M
    y0, y1 = iy.min() * CELL_M, (iy.max() + 1) * CELL_M
    z0, z1 = iz.min() * CELL_M, (iz.max() + 1) * CELL_M
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    return go.Mesh3d(x=x, y=y, z=z,
                     i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
                     j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
                     k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                     color="royalblue", opacity=0.85, flatshading=True,
                     name="True prism", showlegend=False,
                     hovertemplate="x=%{x:.0f} m<br>y=%{y:.0f} m<br>depth=%{z:.0f} m<extra>True prism</extra>")


def plot_sample_3d(density, gravity, output_path, reconstruction=None,
                   reconstruction_threshold=0.25, make_gif=True, gif_frames=80,
                   gif_duration_seconds=8.0):
    """Write a 3D density plot with an optional reconstruction.

    Parameters
    ----------
    density : array (nz,ny,nx), g/cm^3; one uniform rectangular prism
    gravity : retained for compatibility; not displayed
    output_path : path to .html or .gif
    reconstruction : optional array (nz,ny,nx), adjoint/CNN image
    reconstruction_threshold : fraction of maximum positive reconstruction value
    """
    density = np.asarray(density)
    if density.ndim != 3:
        raise ValueError("Expected a 3D density array")
    if not np.isfinite(density).all():
        raise ValueError("Input arrays must contain only finite values")
    nz, ny, nx = density.shape
    if reconstruction is not None:
        reconstruction = np.asarray(reconstruction)
        if reconstruction.shape != density.shape or not np.isfinite(reconstruction).all():
            raise ValueError("Reconstruction must be finite and match density shape")
        if not (0 < reconstruction_threshold < 1):
            raise ValueError("reconstruction_threshold must be between 0 and 1")

    titles = ["True density prism"]
    specs = [[{"type": "scene"}]]
    if reconstruction is not None:
        titles.append("Reconstruction (relative amplitude)")
        specs[0].append({"type": "scene"})

    fig = make_subplots(rows=1, cols=len(titles), specs=specs,
                        subplot_titles=titles, horizontal_spacing=0.06)
    fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=1)
    fig.add_trace(_prism_mesh(density), row=1, col=1)

    if reconstruction is not None:
        peak = float(np.max(reconstruction))
        if peak > 0:
            z, y, x = np.indices(reconstruction.shape)
            fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=2)
            fig.add_trace(go.Isosurface(
                x=((x + 0.5) * CELL_M).ravel(),
                y=((y + 0.5) * CELL_M).ravel(),
                z=((z + 0.5) * CELL_M).ravel(),
                value=(reconstruction / peak).ravel(),
                isomin=reconstruction_threshold, isomax=1,
                surface_count=3, colorscale="Plasma", opacity=0.55,
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                hovertemplate="Relative amplitude=%{value:.3f}<extra></extra>"),
                row=1, col=2)
        else:
            fig.add_trace(_domain_edges(nx, ny, nz), row=1, col=2)

    scene_style = dict(
        xaxis=dict(title="East (m)", range=[0, nx * CELL_M]),
        yaxis=dict(title="North (m)", range=[0, ny * CELL_M]),
        zaxis=dict(title="Depth (m)", range=[nz * CELL_M, 0]),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.55),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    fig.update_layout(scene=scene_style, title="Synthetic density model",
                      width=1300 if reconstruction is not None else 850, height=650,
                      margin=dict(l=15, r=30, t=80, b=25))
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
    """Save a rotating Plotly 3D figure as an animated GIF.

    Uses Plotly/Kaleido for each frame and Pillow to combine frames. The
    The 3D view rotates 360 degrees over the requested duration.
    """
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
                "GIF rendering requires Kaleido and a compatible Chrome/Chromium "
                "installation. Install with: pip install kaleido pillow; "
                "then, if Chrome is missing, run: plotly_get_chrome"
            ) from exc
        with Image.open(BytesIO(png)) as img:
            images.append(img.convert("RGB"))
        print(f"Rendered frame {frame + 1}/{frames}", flush=True)

    images[0].save(output_path, save_all=True, append_images=images[1:],
               duration=round(duration_seconds * 1000 / frames), loop=0,
               disposal=2, optimize=False)
    return output_path


def save_sample_3d(sample_name: str, density: np.ndarray, gravity: np.ndarray,
                   *, root: str | Path = "results/figures",
                   reconstruction: np.ndarray | None = None,
                   make_gif: bool = True,
                   gif_frames: int = 80,
                   gif_duration_seconds: float = 8.0) -> Path:
    """Save the animated Plotly view inside a sample-named subfolder.

    Example output:
        results/figures/sample_000000/interactive_3d.gif
    """
    sample_dir = Path(root) / str(sample_name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    gif_path = sample_dir / "interactive_3d.gif"
    html_path = sample_dir / "interactive_3d.html"

    plot_sample_3d(density, gravity, gif_path if make_gif else html_path,
                   reconstruction=reconstruction,
                   make_gif=make_gif,
                   gif_frames=gif_frames,
                   gif_duration_seconds=gif_duration_seconds)
    if make_gif:
        return gif_path
    return html_path

