"""Compare saved ML density predictions with observed surface gravity.

Run from the repository root:
    python -m src.physics.gravity_consistency --name E06B --component gz
    python -m src.physics.gravity_consistency --name E07A --component gzz
    python -m src.physics.gravity_consistency --name E06B --limit 4

Uses the project's existing forward operator, not TensorFlow. Reads prediction
NPZ files under outputs/ml/predict/<name>/**/predictions/<split>/.
Creates one three-panel map per geometry, a pooled residual histogram, and
per-sample/per-geometry CSVs. Gz is in mGal and Gzz is in Eotvos (E).
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
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pooled, bins=70, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", label="Zero residual")
    ax.set(xlabel=f"Observed - modeled {component.upper()} ({unit})",
           ylabel="Observation-cell count",
           title=f"{args.name}: {component.upper()} residuals ({len(rows)} {args.split} samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "gravity_residuals.png", dpi=args.dpi)
    plt.close(fig)
    print(f"\nSaved figures and CSV files to {output}")
    for item in summary:
        print(f"{item['geometry']}: n={item['n_samples']}, "
              f"mean RMSE={item['mean_rmse']:.5g} {unit}")


if __name__ == "__main__":
    main()
