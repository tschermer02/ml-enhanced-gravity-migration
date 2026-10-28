"""Evaluate E02 predictions and independently recompute gravity with NumPy."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for import_path in (str(SRC), str(SCRIPTS)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from gravity_migration_ml.ml.e02_dataset import make_e02_dataset, load_e02_sample  # noqa: E402
from physics.gravity_forward import OBS_X, OBS_Y, calculate_gravity  # noqa: E402
from visualization.plotly_3d import plot_sample_3d  # noqa: E402
from predict_unet import calculate_metrics, save_comparison_figure  # noqa: E402

TRAINING_ROOT = ROOT / "outputs" / "ml" / "training" / "E02"
E01_TRAINING_ROOT = ROOT / "outputs" / "ml" / "training" / "E01"
E01_PREDICTION_ROOT = ROOT / "outputs" / "ml" / "predict" / "E01"
E02_PREDICTION_ROOT = ROOT / "outputs" / "ml" / "predict" / "E02"
E01_MIGRATION_ROOT = E01_TRAINING_ROOT / "data"
GRAVITY_ROOT = SRC / "data" / "dataset" / "forward_models"
DENSITY_THRESHOLD = 0.1
CELL_SIZE_M = 10.0


def plot_gravity_comparison(observed: np.ndarray, predicted: np.ndarray,
                            output: Path, sample_id: str) -> None:
    """Plot observed Gz, NumPy-predicted Gz, and their residual in mGal."""
    residual = predicted - observed
    shared_min = min(float(observed.min()), float(predicted.min()))
    shared_max = max(float(observed.max()), float(predicted.max()))
    residual_limit = max(float(np.max(np.abs(residual))), 1e-12)
    extent = [float(OBS_X[0]), float(OBS_X[-1]), float(OBS_Y[0]), float(OBS_Y[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = (
        (observed, "Observed gravity", "viridis", shared_min, shared_max),
        (predicted, "Gravity from predicted density", "viridis", shared_min, shared_max),
        (residual, "Predicted - observed residual", "RdBu_r", -residual_limit, residual_limit),
    )
    for axis, (values, title, cmap, low, high) in zip(axes, panels):
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap,
                            vmin=low, vmax=high, aspect="equal")
        axis.set(title=title, xlabel="Easting (m)", ylabel="Northing (m)")
        fig.colorbar(image, ax=axis, label="Gz (mGal)")
    fig.suptitle(f"E02 gravity consistency: {sample_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_e01_e02_comparison(e02_rows: list[dict[str, object]], output: Path) -> None:
    """Join per-sample E01 and E02 test metrics into a comparison CSV."""
    e01_csv = E01_PREDICTION_ROOT / "predictions" / "test_metrics.csv"
    if e01_csv.is_file():
        with e01_csv.open(newline="", encoding="utf-8") as stream:
            e01_rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    else:
        e01_rows = {}
        print(f"E01 metrics not found; E01 columns will be blank: {e01_csv}", flush=True)

    comparison_fields = [
        "sample_id", "e01_density_mae", "e02_density_mae",
        "e01_density_rmse", "e02_density_rmse",
        "e01_density_mae_inside_true_body", "e02_density_mae_inside_true_body",
        "e01_body_iou", "e02_body_iou", "e01_body_dice", "e02_body_dice",
        "e01_top_depth_error_m", "e02_top_depth_error_m",
        "e01_bottom_depth_error_m", "e02_bottom_depth_error_m",
        "e01_thickness_error_m", "e02_thickness_error_m",
        "e02_gravity_mae_mgal", "e02_gravity_rmse_mgal", "e02_gravity_relative_l2",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        for e02 in e02_rows:
            e01 = e01_rows.get(str(e02["sample_id"]), {})
            record: dict[str, object] = {"sample_id": e02["sample_id"]}
            common = (
                "density_mae", "density_rmse", "density_mae_inside_true_body",
                "body_iou", "body_dice", "top_depth_error_m",
                "bottom_depth_error_m", "thickness_error_m",
            )
            for metric in common:
                record[f"e01_{metric}"] = e01.get(metric, "")
                record[f"e02_{metric}"] = e02.get(metric, "")
            for metric in ("gravity_mae_mgal", "gravity_rmse_mgal", "gravity_relative_l2"):
                record[f"e02_{metric}"] = e02.get(metric, "")
            writer.writerow(record)
    print(f"E01 vs E02 comparison: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--data-root", type=Path, default=E01_MIGRATION_ROOT)
    parser.add_argument("--gravity-root", type=Path, default=GRAVITY_ROOT)
    parser.add_argument("--model", type=Path,
                        default=TRAINING_ROOT / "checkpoints" / "best_unet3d.keras")
    parser.add_argument("--output-root", type=Path, default=E02_PREDICTION_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visualize-count", type=int, default=3)
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--no-gifs", action="store_true",
                        help="Save interactive HTML but skip rotating GIF rendering.")
    args = parser.parse_args()
    if args.visualize_count < 0 or args.gif_frames < 2:
        parser.error("--visualize-count must be >= 0 and --gif-frames must be >= 2")

    resolve = lambda path: path if path.is_absolute() else ROOT / path
    data_root, gravity_root = resolve(args.data_root), resolve(args.gravity_root)
    model_path, output_root = resolve(args.model), resolve(args.output_root)
    if not model_path.is_file():
        raise FileNotFoundError(f"E02 checkpoint not found: {model_path}")
    dataset = make_e02_dataset(
        data_root, gravity_root, args.split, batch_size=1, limit=args.limit,
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    prediction_dir = output_root / "predictions" / args.split
    figures_dir = output_root / "figures" / args.split
    prediction_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Keep file ordering aligned with the deterministic dataset loader.
    from gravity_migration_ml.ml.dataset import sample_paths
    files = sample_paths(data_root, args.split, args.limit)
    rows: list[dict[str, object]] = []
    for index, ((migration_input, targets), path) in enumerate(zip(dataset, files), 1):
        density_true = targets["density"].numpy()[0, ..., 0]
        gravity_observed = targets["gravity"].numpy()[0, ..., 0]
        migration_input_volume = migration_input.numpy()[0, ..., 0]
        density_predicted = model(migration_input, training=False).numpy()[0, ..., 0]
        if not np.isfinite(density_predicted).all():
            raise ValueError(f"{path.name}: E02 model produced nonfinite density")

        # Independent evaluation through the original NumPy implementation.
        gravity_predicted = calculate_gravity(density_predicted.astype(np.float64))
        gravity_residual = gravity_predicted - gravity_observed
        gravity_norm = float(np.linalg.norm(gravity_observed))
        metrics = calculate_metrics(density_true, density_predicted, DENSITY_THRESHOLD)
        metrics.update({
            "gravity_mae_mgal": float(np.mean(np.abs(gravity_residual))),
            "gravity_rmse_mgal": float(np.sqrt(np.mean(np.square(gravity_residual)))),
            "gravity_relative_l2": (
                float(np.linalg.norm(gravity_residual) / gravity_norm) if gravity_norm else 0.0
            ),
        })
        row: dict[str, object] = {"sample_id": path.stem, **metrics}
        rows.append(row)
        with np.load(path) as archive:
            raw_migration = np.asarray(archive["migration"], dtype=np.float32)
            migration_scale = float(archive["migration_scale"])
        np.savez_compressed(
            prediction_dir / path.name,
            prediction=density_predicted.astype(np.float32),
            density=density_true.astype(np.float32),
            migration=raw_migration,
            migration_input=migration_input_volume.astype(np.float32),
            migration_scale=np.float64(migration_scale),
            gravity_observed=gravity_observed.astype(np.float32),
            gravity_predicted=gravity_predicted.astype(np.float32),
            gravity_residual=gravity_residual.astype(np.float32),
        )

        if index <= args.visualize_count:
            save_comparison_figure(
                density_true, density_predicted,
                figures_dir / f"{path.stem}_density_comparison.png", path.stem,
            )
            plot_gravity_comparison(
                gravity_observed, gravity_predicted,
                figures_dir / f"{path.stem}_gravity_comparison.png", path.stem,
            )
            try:
                plot_sample_3d(
                    density_true, gravity_observed,
                    figures_dir / f"{path.stem}_density_3d.html",
                    reconstruction=density_predicted,
                    make_gif=not args.no_gifs,
                    gif_frames=args.gif_frames,
                    gif_duration_seconds=8.0,
                )
            except (RuntimeError, ValueError) as error:
                print(
                    f"Optional 3D/GIF export skipped for {path.stem}; "
                    f"prediction and 2D outputs will continue: {error}",
                    flush=True,
                )
        print(
            f"E02 {index}/{len(files)} {path.stem}: density MAE="
            f"{metrics['density_mae']:.5g}, gravity RMSE="
            f"{metrics['gravity_rmse_mgal']:.5g} mGal",
            flush=True,
        )

    metric_fields = list(rows[0].keys())
    metrics_path = output_root / "predictions" / f"{args.split}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)
    numeric_fields = [key for key in metric_fields
                      if key not in {"sample_id", "empty_prediction"}]
    summary = {
        "experiment": "E02",
        "split": args.split,
        "sample_count": len(rows),
        "empty_prediction_count": sum(bool(row["empty_prediction"]) for row in rows),
        "density_threshold_gcm3": DENSITY_THRESHOLD,
        "metrics": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in rows])),
                "std": float(np.std([float(row[key]) for row in rows])),
            }
            for key in numeric_fields
        },
    }
    summary_path = output_root / "predictions" / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.split == "test":
        write_e01_e02_comparison(
            rows, output_root / "predictions" / "E01_vs_E02_test_metrics.csv",
        )
    print(json.dumps(summary, indent=2))
    print(f"Metrics: {metrics_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
