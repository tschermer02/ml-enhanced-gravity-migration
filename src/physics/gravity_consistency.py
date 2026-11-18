"""Compare saved ML density predictions with observed surface gravity.

Run from the repository root:
    python -m src.physics.gravity_consistency --name E06B --component gz
    python -m src.physics.gravity_consistency --name E07A --component gzz
    python -m src.physics.gravity_consistency --name E06B --component gz --limit 4

Uses the project's existing SciPy gravity forward operator, not TensorFlow.
Reads prediction NPZ files under outputs/ml/predict/<name>/**/predictions/<split>/.
Creates one three-panel gravity comparison per geometry, a combined
comparison figure for all geometries, pooled residual plots, and CSVs. Gz units: mGal; Gzz units: Eotvos (E).
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



def save_combined_comparisons(representatives: dict, component: str,
                              unit: str, destination: Path, dpi: int) -> None:
    """Plot one representative test sample per geology in a 4-by-3 layout.

    Columns: observed, forward modeled from ML density, observed-minus-modeled.
    Observed and modeled panels use one shared color scale across all rows;
    residual panels use a separate shared scale across all rows. This makes
    between-category comparisons meaningful.
    """
    if not representatives:
        return

    order = ["compact", "single", "single_compact", "single_compact_body",
             "multiple", "multiple_depths", "multiple_compact_bodies",
             "dipping", "dipping_elongated", "dipping_elongated_body",
             "salt_dome", "salt", "salt_dome_analogue"]
    keys = [key for key in order if key in representatives]
    keys += sorted(key for key in representatives if key not in keys)

    titles = {
        "compact": "A — Single compact body",
        "single": "A — Single compact body",
        "single_compact": "A — Single compact body",
        "single_compact_body": "A — Single compact body",
        "multiple": "B — Multiple compact bodies",
        "multiple_depths": "B — Multiple compact bodies",
        "multiple_compact_bodies": "B — Multiple compact bodies",
        "dipping": "C — Dipping elongated body",
        "dipping_elongated": "C — Dipping elongated body",
        "dipping_elongated_body": "C — Dipping elongated body",
        "salt": "D — Salt-dome analogue",
        "salt_dome": "D — Salt-dome analogue",
        "salt_dome_analogue": "D — Salt-dome analogue",
    }
    field_limit = max(
        max(float(np.max(np.abs(representatives[key][field])))
            for key in keys for field in ("observed", "modeled")), 1e-12)
    residual_limit = max(
        max(float(np.max(np.abs(representatives[key]["residual"])))
            for key in keys), 1e-12)

    nrows = len(keys)
    fig, axes = plt.subplots(
        nrows, 3, figsize=(13.0, 3.0 * nrows + 0.7),
        constrained_layout=True, squeeze=False)
    headings = ("Observed", "Forward-modeled from ML density",
                "Residual (observed − modeled)")
    for j, title in enumerate(headings):
        axes[0, j].set_title(title, fontsize=11, pad=10)

    field_image = None
    residual_image = None
    extent = [OBS_X[0], OBS_X[-1], OBS_Y[0], OBS_Y[-1]]

    for i, key in enumerate(keys):
        sample = representatives[key]
        residual = sample["residual"]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        for j, field_name in enumerate(("observed", "modeled", "residual")):
            ax = axes[i, j]
            limit = field_limit if j < 2 else residual_limit
            im = ax.imshow(
                sample[field_name], origin="lower", cmap="RdBu_r",
                extent=extent,
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                interpolation="nearest", aspect="equal")
            if j < 2:
                field_image = im
            else:
                residual_image = im
            ax.set_xlabel("Easting (m)", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{titles.get(key, key.replace('_', ' ').title())}\nNorthing (m)",
                              fontsize=9)
            else:
                ax.set_ylabel("Northing (m)", fontsize=9)
            ax.tick_params(labelsize=8)
            if j == 2:
                ax.text(0.97, 0.96, f"RMSE: {rmse:.3g} {unit}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9,
                                  boxstyle="round,pad=0.25"))

    cb_field = fig.colorbar(field_image, ax=axes[:, :2].ravel().tolist(),
                            shrink=0.84, pad=0.015)
    cb_field.set_label(f"{component.upper()} ({unit})")
    cb_resid = fig.colorbar(residual_image, ax=axes[:, 2].ravel().tolist(),
                            shrink=0.84, pad=0.015)
    cb_resid.set_label(f"Residual {component.upper()} ({unit})")
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
    representatives = {}
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
            representatives[geometry] = {
                "observed": observed.copy(),
                "modeled": modeled.copy(),
                "residual": residual.copy(),
                "sample": path.stem,
            }
        print(f"{i}/{len(paths)} {geometry} {path.stem}: RMSE={rmse:.5g} {unit}", flush=True)

    component = next(iter(units_seen))
    unit = "mGal" if component == "gz" else "E"
    save_combined_comparisons(
        representatives, component, unit,
        output / "gravity_comparison_all_geometries.png", args.dpi)

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

# -----------------------------------------------------------------------------
# E11: optional physical-grid / elevated-receiver consistency analysis.
# Keep main() above unchanged for all legacy E01-E10 commands.
# -----------------------------------------------------------------------------
def main_e11():
    """Compare E11 prediction Gzz with measured and clean synthetic Gzz.

    Uses the sample-specific cell dimensions and height. Retains all original
    histogram/plot functions, but supplies kilometer axes for E11 figures.
    """
    import json
    from collections import defaultdict
    from src.physics.gravity_forward import observation_coordinates

    parser = argparse.ArgumentParser(description="E11 Gzz gravity-consistency analysis")
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--hist-percentile", type=float, default=99.5)
    parser.add_argument("--hist-bins", type=int, default=60)
    parser.add_argument("--reference", choices=("observed", "clean"), default="observed",
                        help="Observed noisy Gzz (default) or clean synthetic Gzz")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 < args.hist_percentile <= 100:
        parser.error("--hist-percentile must be in (0,100]")
    if args.hist_bins < 2 or args.hist_bins % 2:
        parser.error("--hist-bins must be even and >= 2")

    def resolve(value):
        value = Path(value)
        return value if value.is_absolute() else ROOT / value

    source = resolve(args.predictions_root or (ROOT / "outputs/ml/predict" / args.name))
    output = resolve(args.output or source / "gravity_consistency" / args.split)
    paths = sorted(source.glob(f"**/predictions/{args.split}/*.npz"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"No prediction NPZ files for {args.split} in {source}")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    all_residuals = []
    representative = {}
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as f:
            target_key = "gravity_clean" if args.reference == "clean" else "gravity"
            required = {"prediction", target_key, "dx_m", "dy_m", "dz_m",
                        "observation_height_m"}
            missing = required.difference(f.files)
            if missing:
                raise KeyError(f"{path}: missing required arrays {sorted(missing)}")
            density = np.asarray(f["prediction"], dtype=np.float64).squeeze()
            observed = np.asarray(f[target_key], dtype=np.float64).squeeze()
            dx = float(np.asarray(f["dx_m"]).item())
            dy = float(np.asarray(f["dy_m"]).item())
            dz = float(np.asarray(f["dz_m"]).item())
            height = float(np.asarray(f["observation_height_m"]).item())
            geometry = str(f["geometry"].item()).lower() if "geometry" in f else "unknown"
            noise = float(f["noise_percent"].item()) if "noise_percent" in f else 0.0
            component = (str(f["observation_component"].item()).lower()
                         if "observation_component" in f else "gzz")
        if component != "gzz":
            raise ValueError(f"{path}: E11 analysis requires Gzz, got {component}")
        if density.shape != SHAPE or observed.shape != OBS_SHAPE:
            raise ValueError(f"{path}: expected {SHAPE} density and {OBS_SHAPE} Gzz")
        if not np.isfinite(density).all() or not np.isfinite(observed).all():
            raise ValueError(f"{path}: nonfinite density or Gzz")
        if not all(np.isfinite((dx, dy, dz, height))) or min(dx, dy, dz) <= 0 or height < 0:
            raise ValueError(f"{path}: invalid physical grid or receiver height")

        modeled = np.asarray(calculate_gravity(
            density, component="gzz", dx_m=dx, dy_m=dy, dz_m=dz,
            observation_height_m=height), dtype=np.float64)
        residual = observed - modeled
        rmse = float(np.sqrt(np.mean(residual**2)))
        mae = float(np.mean(np.abs(residual)))
        bias = float(np.mean(residual))
        denominator = float(np.std(observed) * np.std(modeled))
        correlation = (float(np.corrcoef(observed.ravel(), modeled.ravel())[0, 1])
                       if denominator > 0 else float("nan"))
        rows.append(dict(sample=path.stem, geometry=geometry, component="gzz",
                         reference=args.reference, noise_percent=noise,
                         observation_height_m=height, dx_m=dx, dy_m=dy, dz_m=dz,
                         rmse_e=rmse, mae_e=mae, mean_residual_e=bias,
                         correlation=correlation))
        all_residuals.append(residual.ravel())
        representative.setdefault(geometry, (observed, modeled, residual, dx, dy, height))
        print(f"{index}/{len(paths)} {geometry} {path.stem} | height={height:g} m | "
              f"RMSE={rmse:.5g} E", flush=True)

    fields = list(rows[0])
    save_csv(output / "gravity_metrics_per_sample.csv", rows, fields)

    def statistics(group):
        def describe(key):
            vals = np.asarray([r[key] for r in group], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else None
        return dict(n_samples=len(group), mean_rmse_e=describe("rmse_e"),
                    mean_mae_e=describe("mae_e"), mean_bias_e=describe("mean_residual_e"),
                    mean_correlation=describe("correlation"))

    groups = defaultdict(list)
    geometry_groups = defaultdict(list)
    for row in rows:
        groups[(row["geometry"], row["noise_percent"], row["observation_height_m"])].append(row)
        geometry_groups[row["geometry"]].append(row)
    summary = {
        "split": args.split,
        "reference": args.reference,
        "unit": "E",
        "overall": statistics(rows),
        "by_geometry": {g: statistics(v) for g, v in sorted(geometry_groups.items())},
        "by_geometry_noise_height": {
            f"{geom}|noise_{noise:g}|height_{height:g}m": statistics(group)
            for (geom, noise, height), group in sorted(groups.items())
        },
    }
    (output / "gravity_summary.json").write_text(json.dumps(summary, indent=2))

    # Preserve the existing grouped CSV convention.
    by_geom = [dict(geometry=geom, **statistics(group))
               for geom, group in sorted(geometry_groups.items())]
    save_csv(output / "gravity_metrics_by_geometry.csv", by_geom,
             ["geometry", "n_samples", "mean_rmse_e", "mean_mae_e",
              "mean_bias_e", "mean_correlation"])

    for geom, (observed, modeled, residual, dx, dy, height) in representative.items():
        obs_x, obs_y = observation_coordinates(dx_m=dx, dy_m=dy)
        extent = [obs_x[0]/1000, obs_x[-1]/1000,
                  obs_y[0]/1000, obs_y[-1]/1000]
        scale = max(np.max(np.abs(observed)), np.max(np.abs(modeled)), 1e-12)
        residual_scale = max(np.max(np.abs(residual)), 1e-12)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for ax, data, name, limit in zip(
            axes, (observed, modeled, residual),
            (f"{args.reference.capitalize()} Gzz", "Predicted-density Gzz", "Residual"),
            (scale, scale, residual_scale)
        ):
            image = ax.imshow(data, origin="lower", cmap="RdBu_r", extent=extent,
                              norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
            ax.set(xlabel="Easting (km)", ylabel="Northing (km)", title=name)
            fig.colorbar(image, ax=ax, label="Gzz (E)", shrink=.8)
        fig.suptitle(f"{geom}: receiver height {height:g} m")
        fig.savefig(output / f"gravity_comparison_{geom}.png", dpi=args.dpi,
                    bbox_inches="tight")
        plt.close(fig)

    pooled = np.concatenate(all_residuals)
    save_residual_histograms(pooled, "gzz", "E", output, args.dpi,
                             percentile=args.hist_percentile, bins=args.hist_bins)
    print(f"E11 gravity-consistency results saved to {output}", flush=True)


if __name__ == "__main__":
    import sys
    if "--e11" in sys.argv:
        sys.argv.remove("--e11")
        main_e11()
    else:
        main()
