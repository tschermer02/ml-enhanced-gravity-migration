"""Compare saved ML density predictions with observed surface gravity.

Run from the repository root:
    python -m src.physics.gravity_consistency --name E06B --component gz
    python -m src.physics.gravity_consistency --name E07A --component gzz
    python -m src.physics.gravity_consistency --name E06B --component gz --limit 4

Uses the project's existing SciPy gravity forward operator, not TensorFlow.
Reads prediction NPZ files under outputs/ml/predict/<name>/**/predictions/<split>/.
Creates one three-panel gravity comparison per geometry, pooled residual plots,
and per-sample/per-geometry CSVs. Gz units: mGal; Gzz units: Eotvos (E).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from src.physics.gravity_forward import calculate_gravity, OBS_X, OBS_Y, SHAPE, OBS_SHAPE

ROOT = Path(__file__).resolve().parents[2]


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(observed, modeled, residual, geometry, sample_id,
                    component, unit, destination, dpi):
    """Keep existing full-field three-panel comparison plots."""
    shared = max(float(np.max(np.abs(observed))),
                 float(np.max(np.abs(modeled))), 1e-12)
    res_scale = max(float(np.max(np.abs(residual))), 1e-12)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for index, (ax, field, title) in enumerate(zip(
        axes, (observed, modeled, residual),
        ("Observed", "Forward-modeled from ML density", "Residual (observed - modeled)"),
    )):
        limit = res_scale if index == 2 else shared
        im = ax.imshow(field, origin="lower", cmap="RdBu_r",
                       extent=[OBS_X[0], OBS_X[-1], OBS_Y[0], OBS_Y[-1]],
                       norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        ax.set(title=title, xlabel="Easting (m)", ylabel="Northing (m)")
        fig.colorbar(im, ax=ax, label=f"{component.upper()} ({unit})", shrink=0.8)
    fig.suptitle(f"{geometry}: {sample_id} | {component.upper()} RMSE = {rmse:.5g} {unit}")
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def centered_bins(limit: float, bin_count: int) -> np.ndarray:
    """Symmetric, evenly spaced edges; use an even number of bins."""
    if bin_count < 2 or bin_count % 2:
        raise ValueError("Histogram bin count must be positive and even (at least 2)")
    return np.linspace(-limit, limit, bin_count + 1)


def style_histogram_axis(ax, component: str, unit: str) -> None:
    ax.axvline(0, color="black", linestyle="--", linewidth=1.3,
               label="Zero residual", zorder=3)
    label = rf"$G_{{{'z' if component == 'gz' else 'zz'}}}^{{\mathrm{{obs}}}}"
    label += rf" - G_{{{'z' if component == 'gz' else 'zz'}}}^{{\mathrm{{pred}}}}$ ({unit})"
    ax.set_xlabel(label)
    ax.set_ylabel("Observations per bin (%)")
    ax.grid(axis="y", alpha=0.20, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_histogram(ax, residuals: np.ndarray, edges: np.ndarray, weights: np.ndarray):
    ax.hist(residuals, bins=edges, weights=weights,
            color="#4477AA", edgecolor="white", linewidth=0.45)
    ax.set_xlim(edges[0], edges[-1])


def save_residual_histograms(pooled: np.ndarray, component: str, unit: str,
                             destination: Path, dpi: int,
                             percentile: float = 99.5, bins: int = 60) -> None:
    """Create zoomed paper figure and full/zoom side-by-side view.

    Percentages always use the *complete* number of observation cells as
    denominator; observations outside the zoom range are not renormalized.
    RMSE, MAE, and bias are computed on all residuals, including outliers.
    """
    values = np.asarray(pooled, dtype=np.float64).ravel()
    if not np.isfinite(values).all() or values.size == 0:
        raise ValueError("Residuals must be nonempty and finite")

    rmse = float(np.sqrt(np.mean(values ** 2)))
    mae = float(np.mean(np.abs(values)))
    bias = float(np.mean(values))
    # Zoom limit based on absolute residuals, preserving a symmetric x-axis.
    zoom_limit = float(np.percentile(np.abs(values), percentile))
    full_limit = float(np.max(np.abs(values)))
    zoom_limit = max(zoom_limit, 1e-12)
    full_limit = max(full_limit, 1e-12)
    zoom_edges = centered_bins(zoom_limit, bins)
    full_edges = centered_bins(full_limit * (1 + 1e-9), bins)
    weights = np.full(values.shape, 100.0 / values.size)
    coverage = float(np.mean(np.abs(values) <= zoom_limit) * 100.0)

    # Paper-ready single-panel histogram.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    draw_histogram(ax, values, zoom_edges, weights)
    style_histogram_axis(ax, component, unit)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.96,
            f"RMSE = {rmse:.4g} {unit}\nMAE = {mae:.4g} {unit}\n"
            f"Mean = {bias:.4g} {unit}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.85",
                      boxstyle="round,pad=0.35", alpha=0.95))
    ax.text(0.02, -0.21, f"Central {coverage:.2f}% of residuals shown; "
            f"metrics use all {values.size:,} observations.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="0.35")
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(destination / "gravity_residuals.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Additional two-panel histogram: preserves full-range outliers.
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), sharey=False)
    for ax, edges, title in zip(
        axes, (full_edges, zoom_edges),
        ("Full residual range", f"Central {coverage:.2f}% (zoom)"),
    ):
        draw_histogram(ax, values, edges, weights)
        style_histogram_axis(ax, component, unit)
        ax.set_title(title, fontsize=11)
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].text(0.97, 0.96,
                 f"RMSE = {rmse:.4g} {unit}\nMAE = {mae:.4g} {unit}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="0.85",
                           boxstyle="round,pad=0.35", alpha=0.95))
    fig.tight_layout()
    fig.savefig(destination / "gravity_residuals_full_and_zoom.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Experiment name (e.g., E06B)")
    parser.add_argument("--split", default="test", choices=("test", "validation", "train"))
    parser.add_argument("--predictions-root", type=Path, default=None,
                        help="Override experiment prediction folder")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output folder")
    parser.add_argument("--component", choices=("gz", "gzz"), default=None,
                        help="Override component metadata (use for older files)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files for testing")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hist-percentile", type=float, default=99.5,
                        help="Central absolute-residual percentile shown in zoom figure")
    parser.add_argument("--hist-bins", type=int, default=60,
                        help="Even number of histogram bins (default: 60)")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0.0 < args.hist_percentile <= 100.0:
        parser.error("--hist-percentile must be in (0, 100]")
    if args.hist_bins < 2 or args.hist_bins % 2:
        parser.error("--hist-bins must be even and at least 2")

    predictions_root = args.predictions_root or (ROOT / "outputs/ml/predict" / args.name)
    if not predictions_root.is_absolute():
        predictions_root = ROOT / predictions_root
    paths = sorted(predictions_root.glob(f"**/predictions/{args.split}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No prediction NPZ files found under {predictions_root} for {args.split}")
    if args.limit is not None:
        paths = paths[:args.limit]

    output = args.output or (predictions_root / "gravity_consistency" / args.split)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    residual_blocks = []
    plotted = set()
    units_seen = set()

    for i, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as sample:
            missing = {"gravity", "prediction"} - set(sample.files)
            if missing:
                raise KeyError(f"{path}: missing required arrays {sorted(missing)}")
            density = np.asarray(sample["prediction"], dtype=np.float64).squeeze()
            observed = np.asarray(sample["gravity"], dtype=np.float64).squeeze()
            geometry = (str(sample["geometry"].item()).lower() if "geometry" in sample.files
                        else path.parent.parent.parent.name)
            stored_component = next((str(sample[key].item()).lower()
                                     for key in ("observation_component", "observed_component")
                                     if key in sample.files), None)
        component = args.component or stored_component
        if component is None:
            raise ValueError(f"{path}: missing observation component; pass --component gz or gzz")
        if component not in ("gz", "gzz"):
            raise ValueError(f"{path}: unexpected component {component!r}")
        if args.component and stored_component and stored_component != args.component:
            raise ValueError(f"{path}: stored component {stored_component!r} conflicts "
                             f"with --component {args.component!r}; refusing to mix units")
        if density.shape != SHAPE or observed.shape != OBS_SHAPE:
            raise ValueError(f"{path}: expected density {SHAPE} and gravity {OBS_SHAPE}; "
                             f"found {density.shape} and {observed.shape}")
        if not np.isfinite(density).all() or not np.isfinite(observed).all():
            raise ValueError(f"{path}: nonfinite values in density or gravity")

        unit = "mGal" if component == "gz" else "E"
        units_seen.add(component)
        if len(units_seen) > 1:
            raise ValueError("Mixed Gz/Gzz files detected; evaluate each component separately")
        modeled = np.asarray(calculate_gravity(density, component=component), dtype=np.float64)
        residual = observed - modeled
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        bias = float(np.mean(residual))
        rows.append({"sample": path.stem, "geometry": geometry, "component": component,
                     "rmse": rmse, "mae": mae, "mean_residual": bias})
        residual_blocks.append(residual.ravel())

        if geometry not in plotted:
            save_comparison(observed, modeled, residual, geometry, path.stem,
                            component, unit, output / f"gravity_comparison_{geometry}.png", args.dpi)
            plotted.add(geometry)
        print(f"{i}/{len(paths)} {geometry} {path.stem}: RMSE={rmse:.5g} {unit}", flush=True)

    save_csv(output / "gravity_metrics_per_sample.csv", rows,
             ["sample", "geometry", "component", "rmse", "mae", "mean_residual"])
    summary = []
    for geometry in sorted({row["geometry"] for row in rows}):
        group = [row for row in rows if row["geometry"] == geometry]
        summary.append({"geometry": geometry, "component": next(iter(units_seen)),
                        "n_samples": len(group),
                        "mean_rmse": float(np.mean([r["rmse"] for r in group])),
                        "std_rmse": float(np.std([r["rmse"] for r in group])),
                        "mean_mae": float(np.mean([r["mae"] for r in group]))})
    save_csv(output / "gravity_metrics_by_geometry.csv", summary,
             ["geometry", "component", "n_samples", "mean_rmse", "std_rmse", "mean_mae"])

    component = next(iter(units_seen))
    unit = "mGal" if component == "gz" else "E"
    pooled = np.concatenate(residual_blocks)
    save_residual_histograms(pooled, component, unit, output, args.dpi,
                             percentile=args.hist_percentile, bins=args.hist_bins)
    print(f"\nSaved figures and CSV files to {output}")
    for item in summary:
        print(f"{item['geometry']}: n={item['n_samples']}, "
              f"mean RMSE={item['mean_rmse']:.5g} {unit}")


if __name__ == "__main__":
    main()
