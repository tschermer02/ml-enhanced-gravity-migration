"""Predict test density volumes and evaluate the trained 3D U-Net."""
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gravity_migration_ml.ml.dataset import load_sample, sample_paths  # noqa: E402

TRAINING_ROOT = ROOT / "outputs" / "ml" / "training" / "E01"
DATA_ROOT = TRAINING_ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs" / "ml" / "predict" / "E01"
CELL_SIZE_M = 10.0
DENSITY_THRESHOLD = 0.1
VOLUME_SHAPE = (24, 64, 64)


def calculate_metrics(truth: np.ndarray, prediction: np.ndarray,
                      threshold: float = DENSITY_THRESHOLD) -> dict[str, float | bool]:
    """Calculate volume and body-geometry metrics with finite empty-body scores."""
    difference = prediction - truth
    true_body = truth >= threshold
    predicted_body = prediction >= threshold
    intersection = int(np.count_nonzero(true_body & predicted_body))
    union = int(np.count_nonzero(true_body | predicted_body))
    true_count = int(np.count_nonzero(true_body))
    predicted_count = int(np.count_nonzero(predicted_body))
    empty_prediction = predicted_count == 0
    iou = intersection / union if union else 1.0
    dice_denominator = true_count + predicted_count
    dice = 2.0 * intersection / dice_denominator if dice_denominator else 1.0
    body_mae = float(np.mean(np.abs(difference[true_body]))) if true_count else 0.0

    true_depths = np.flatnonzero(np.any(true_body, axis=(1, 2)))
    if true_depths.size:
        true_top = float(true_depths[0] * CELL_SIZE_M)
        true_bottom = float((true_depths[-1] + 1) * CELL_SIZE_M)
    else:
        true_top = 0.0
        true_bottom = 0.0
    if empty_prediction:
        # A missing body gets a finite, full-domain depth penalty.
        top_error = float(VOLUME_SHAPE[0] * CELL_SIZE_M)
        bottom_error = top_error
        thickness_error = true_bottom - true_top
    else:
        predicted_depths = np.flatnonzero(np.any(predicted_body, axis=(1, 2)))
        predicted_top = float(predicted_depths[0] * CELL_SIZE_M)
        predicted_bottom = float((predicted_depths[-1] + 1) * CELL_SIZE_M)
        top_error = abs(predicted_top - true_top)
        bottom_error = abs(predicted_bottom - true_bottom)
        thickness_error = abs(
            (predicted_bottom - predicted_top) - (true_bottom - true_top)
        )

    return {
        "density_mae": float(np.mean(np.abs(difference))),
        "density_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "density_mae_inside_true_body": body_mae,
        "body_iou": float(iou),
        "body_dice": float(dice),
        "top_depth_error_m": float(top_error),
        "bottom_depth_error_m": float(bottom_error),
        "thickness_error_m": float(thickness_error),
        "empty_prediction": empty_prediction,
    }


def save_comparison_figure(density: np.ndarray, prediction: np.ndarray, output: Path,
                           sample_id: str) -> None:
    """Save matching horizontal and vertical sections for truth and prediction."""
    nz, ny, nx = density.shape
    occupied = np.argwhere(density > 0)
    iz, iy, ix = (
        np.rint(occupied.mean(axis=0)).astype(int)
        if occupied.size else (nz // 2, ny // 2, nx // 2)
    )
    sections = [
        (density[iz], prediction[iz],
         [0, nx * CELL_SIZE_M, 0, ny * CELL_SIZE_M], "lower",
         f"Horizontal, z={(iz + 0.5) * CELL_SIZE_M:g} m", "Easting (m)", "Northing (m)"),
        (density[:, iy, :], prediction[:, iy, :],
         [0, nx * CELL_SIZE_M, nz * CELL_SIZE_M, 0], "upper",
         f"X-depth, y={(iy + 0.5) * CELL_SIZE_M:g} m", "Easting (m)", "Depth (m)"),
        (density[:, :, ix], prediction[:, :, ix],
         [0, ny * CELL_SIZE_M, nz * CELL_SIZE_M, 0], "upper",
         f"Y-depth, x={(ix + 0.5) * CELL_SIZE_M:g} m", "Northing (m)", "Depth (m)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    rows = (
        ("True density", "Density contrast (g/cm³)", 0.0, 1.0, "viridis"),
        ("U-Net prediction", "Density contrast (g/cm³)", 0.0, 1.0, "viridis"),
    )
    for row, (row_label, units, low, high, color_map) in enumerate(rows):
        last_image = None
        for column, (true_slice, predicted_slice,
                     extent, origin, title, xlabel, ylabel) in enumerate(sections):
            values = (true_slice, predicted_slice)[row]
            last_image = axes[row, column].imshow(
                values, origin=origin, extent=extent, cmap=color_map,
                vmin=low, vmax=high, aspect="auto",
            )
            axes[row, column].set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(last_image, ax=axes[row, :], shrink=0.85, label=units)
        axes[row, 0].annotate(
            row_label, xy=(-0.2, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=12,
        )
    fig.suptitle(f"Test sample: {sample_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model", type=Path,
                        default=TRAINING_ROOT / "checkpoints" / "best_unet3d.keras")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--visualize-count", type=int, default=None,
        help="Limit comparison figures to the first N samples; default creates one for every sample.",
    )
    args = parser.parse_args()
    if args.visualize_count is not None and args.visualize_count < 0:
        parser.error("--visualize-count must be >= 0")

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    files = sample_paths(data_root, args.split, args.limit)
    model = tf.keras.models.load_model(model_path, compile=False)
    predictions_dir = output_root / "predictions" / args.split
    figures_dir = output_root / "figures" / args.split
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | bool]] = []
    for index, path in enumerate(files, 1):
        migration_input, true_density = load_sample(path)
        predicted_density = model(migration_input[None, ...], training=False).numpy()[0, ..., 0]
        true_density = true_density[..., 0]
        migration_input = migration_input[..., 0]
        if not np.isfinite(predicted_density).all():
            raise ValueError(f"{path.name}: model produced NaN or infinite predictions")
        if predicted_density.min() < 0 or predicted_density.max() > 1.0:
            raise ValueError(f"{path.name}: sigmoid predictions must be within [0, 1]")

        with np.load(path) as archive:
            migration = np.asarray(archive["migration"], dtype=np.float32)
            migration_scale = float(archive["migration_scale"])
        result_metrics = calculate_metrics(true_density, predicted_density)
        rows.append({"sample_id": path.stem, **result_metrics})
        np.savez_compressed(
            predictions_dir / path.name,
            prediction=predicted_density.astype(np.float32),
            density=true_density.astype(np.float32),
            migration=migration,
            migration_input=migration_input.astype(np.float32),
            migration_scale=np.float64(migration_scale),
        )

        if args.visualize_count is None or index <= args.visualize_count:
            save_comparison_figure(
                true_density, predicted_density,
                figures_dir / f"{path.stem}_comparison.png", path.stem,
            )
        print(f"Predicted {index}/{len(files)}: {path.stem}; "
              f"MAE={result_metrics['density_mae']:.5g}; "
              f"IoU={result_metrics['body_iou']:.4f}; "
              f"empty={result_metrics['empty_prediction']}", flush=True)

    metric_fields = list(rows[0].keys())
    metrics_csv = output_root / "predictions" / f"{args.split}_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = [field for field in metric_fields
                      if field not in {"sample_id", "empty_prediction"}]
    summary = {
        "split": args.split,
        "sample_count": len(rows),
        "empty_prediction_count": sum(bool(row["empty_prediction"]) for row in rows),
        "density_threshold_gcm3": DENSITY_THRESHOLD,
        "empty_prediction_depth_penalty_m": VOLUME_SHAPE[0] * CELL_SIZE_M,
        "metrics": {
            field: {
                "mean": float(np.mean([float(row[field]) for row in rows])),
                "std": float(np.std([float(row[field]) for row in rows])),
            }
            for field in numeric_fields
        },
    }
    summary_path = output_root / "predictions" / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Metrics: {metrics_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
