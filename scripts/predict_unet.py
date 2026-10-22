"""Predict density volumes and evaluate E01 through E05 3D U-Nets.

E01-E03 use the original single-body dataset.
E04 uses the original multi-body dataset.
E04B uses a 2–3-body touching-allowed dataset.
E05 reuses E04 data to isolate the vertical depth-loss change.

Outputs:
  predictions/<split>/*.npz          Predicted and true density volumes
  predictions/<split>_metrics.csv   Per-sample density and geometry metrics
  predictions/<split>_summary.json Mean and std of each metric
  figures/<split>/*_density_comparison.png
  figures/<split>/*_gravity_comparison.png

Optional: --plotly creates GIF-only 3D animations; --no-gifs requests HTML instead.
Use --plots-only to recreate density-section PNGs from saved predictions without inference.
Run from the repository root, for example:
  python scripts/predict_unet.py --experiment E04 --split test --visualize-count 100
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without a graphical desktop (e.g., CHPC).
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gravity_migration_ml.ml.dataset import load_sample, sample_paths
from gravity_migration_ml.ml.e02_physics import gravity_forward_tf

# ============================================================
# 1. CONSTANTS AND EXPERIMENT-SPECIFIC PATHS
# ============================================================

CELL_SIZE_M = 10.0
DENSITY_THRESHOLD = 0.1  # g/cm^3: threshold for binary-body evaluation.
VOLUME_SHAPE = (24, 64, 64)  # z, y, x
GRAVITY_SHAPE = (81, 81)

TRAINING_BASE = ROOT / "outputs" / "ml" / "training"
PREDICTION_BASE = ROOT / "outputs" / "ml" / "predict"
SINGLE_BODY_DATA_ROOT = TRAINING_BASE / "E01" / "data"
MULTI_BODY_DATA_ROOT = TRAINING_BASE / "E04" / "data"
E04B_DATA_ROOT = TRAINING_BASE / "E04B" / "data"
SINGLE_BODY_GRAVITY_ROOT = SRC / "data" / "dataset" / "forward_models"
MULTI_BODY_GRAVITY_ROOT = SRC / "data" / "dataset_multibody" / "forward_models"
E04B_GRAVITY_ROOT = SRC / "data" / "dataset_multibody_e04b" / "forward_models"

EXPERIMENTS = {
    "E01": {
        "training": TRAINING_BASE / "E01",
        "output": PREDICTION_BASE / "E01",
        "data": SINGLE_BODY_DATA_ROOT,
        "gravity": SINGLE_BODY_GRAVITY_ROOT,
    },
    "E02_balanced_huber": {
        "training": TRAINING_BASE / "E02_balanced_huber",
        "output": PREDICTION_BASE / "E02_balanced_huber",
        "data": SINGLE_BODY_DATA_ROOT,
        "gravity": SINGLE_BODY_GRAVITY_ROOT,
    },
    "E03": {
        "training": TRAINING_BASE / "E03",
        "output": PREDICTION_BASE / "E03",
        "data": SINGLE_BODY_DATA_ROOT,
        "gravity": SINGLE_BODY_GRAVITY_ROOT,
    },
    "E04": {
        "training": TRAINING_BASE / "E04",
        "output": PREDICTION_BASE / "E04",
        "data": MULTI_BODY_DATA_ROOT,
        "gravity": MULTI_BODY_GRAVITY_ROOT,
    },
    "E04B": {
        "training": TRAINING_BASE / "E04B",
        "output": PREDICTION_BASE / "E04B",
        "data": E04B_DATA_ROOT,
        "gravity": E04B_GRAVITY_ROOT,
    },
    "E05": {
        "training": TRAINING_BASE / "E05",
        "output": PREDICTION_BASE / "E05",
        "data": MULTI_BODY_DATA_ROOT,  # Same test samples as E04.
        "gravity": MULTI_BODY_GRAVITY_ROOT,
    },
}


def absolute_path(path: Path) -> Path:
    """Interpret relative CLI paths relative to the repository root."""
    return path if path.is_absolute() else ROOT / path


# ============================================================
# 2. DENSITY AND WHOLE-VOLUME GEOMETRY METRICS
# ============================================================

def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    threshold: float = DENSITY_THRESHOLD,
) -> dict[str, float | bool]:
    """Calculate per-volume density/shape metrics for single or multiple bodies.

    IoU/Dice treat all detected bodies as one occupancy mask; they do NOT
    measure the number of distinct bodies recovered.
    """
    difference = prediction - truth
    true_body = truth >= threshold
    predicted_body = prediction >= threshold

    intersection = int(np.count_nonzero(true_body & predicted_body))
    union = int(np.count_nonzero(true_body | predicted_body))
    true_count = int(np.count_nonzero(true_body))
    predicted_count = int(np.count_nonzero(predicted_body))
    empty_prediction = predicted_count == 0

    iou = intersection / union if union else 1.0
    denom = true_count + predicted_count
    dice = 2.0 * intersection / denom if denom else 1.0
    volume_ratio = predicted_count / true_count if true_count else 0.0
    body_mae = float(np.mean(np.abs(difference[true_body]))) if true_count else 0.0

    true_depths = np.flatnonzero(np.any(true_body, axis=(1, 2)))
    if true_depths.size:
        true_top = float(true_depths[0] * CELL_SIZE_M)
        true_bottom = float((true_depths[-1] + 1) * CELL_SIZE_M)
    else:
        true_top = true_bottom = 0.0

    if empty_prediction:
        # Retain the E01-E03 convention for directly comparable metrics.
        top_error = bottom_error = float(VOLUME_SHAPE[0] * CELL_SIZE_M)
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
        "body_volume_ratio": float(volume_ratio),
        "top_depth_error_m": float(top_error),
        "bottom_depth_error_m": float(bottom_error),
        "thickness_error_m": float(thickness_error),
        "empty_prediction": empty_prediction,
    }


# ============================================================
# 3. FIGURE 1: TRUE VS. PREDICTED DENSITY (2 ROWS X 3 COLUMNS)
# ============================================================

def _true_body_components(density: np.ndarray) -> list[np.ndarray]:
    """Return masks for each face-connected true density body.

    With E04, independent prisms can be widely separated, so their overall
    centroid is often outside every body. Six-neighbor connectivity prevents
    distinct prisms touching only at corners from being merged.
    """
    mask = density > 0
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    components, count = label(mask, structure=structure)
    if count == 0:
        return [mask]
    return [components == i for i in range(1, count + 1)]


def save_comparison_figure(
    density: np.ndarray, prediction: np.ndarray, output: Path, sample_id: str,
) -> None:
    """Make three orthogonal slices through EACH true body.

    For each true body the upper row shows ground truth and the lower row the
    network prediction, at identical depth/northing/easting coordinates.
    This fixes blank depth views when multiple true bodies are separated.
    """
    nz, ny, nx = density.shape
    bodies = _true_body_components(density)
    fig, axes = plt.subplots(
        len(bodies) * 2, 3, figsize=(17, max(7, 6 * len(bodies))),
        constrained_layout=True, squeeze=False,
    )
    for body_number, body_mask in enumerate(bodies, start=1):
        coordinates = np.argwhere(body_mask)
        if coordinates.size:
            # A rectangular body's midpoint is inside it. For other shapes,
            # select the occupied voxel closest to the component centroid.
            centroid = coordinates.mean(axis=0)
            nearest = np.argmin(np.sum((coordinates - centroid) ** 2, axis=1))
            iz, iy, ix = (int(v) for v in coordinates[nearest])
        else:
            iz, iy, ix = nz // 2, ny // 2, nx // 2
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
        for local_row, row_title in enumerate(("True density", "U-Net prediction")):
            row = 2 * (body_number - 1) + local_row
            last_image = None
            for column, (true_slice, predicted_slice, extent,
                         origin, title, xlabel, ylabel) in enumerate(sections):
                values = true_slice if local_row == 0 else predicted_slice
                last_image = axes[row, column].imshow(
                    values, origin=origin, extent=extent, cmap="viridis",
                    vmin=0.0, vmax=1.0, aspect="auto",
                )
                axes[row, column].set(
                    title=f"Body {body_number}: {title}",
                    xlabel=xlabel, ylabel=ylabel,
                )
            axes[row, 0].annotate(
                row_title, xy=(-0.20, 0.5), xycoords="axes fraction",
                rotation=90, ha="center", va="center", fontsize=11,
            )
            fig.colorbar(
                last_image, ax=axes[row, :], shrink=0.85,
                label="Density contrast (g/cm³)",
            )
    fig.suptitle(f"Test sample: {sample_id} ({len(bodies)} true bodies)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# 4. FIGURE 2: OBSERVED VS. RECOVERED GRAVITY + RESIDUAL
# ============================================================

def save_gravity_comparison(
    observed: np.ndarray, recovered: np.ndarray, output: Path, sample_id: str,
) -> None:
    """Plot observed/recovered gravity and recovered-minus-observed residual."""
    residual = recovered - observed
    vmax = max(float(np.max(np.abs(observed))), float(np.max(np.abs(recovered))), 1e-10)
    residual_max = max(float(np.max(np.abs(residual))), 1e-10)
    extent = [-85, 715, -85, 715]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, values, title in zip(
        axes, (observed, recovered, residual),
        ("True Gravity", "Recovered Gravity", "Gravity Residual"),
    ):
        if title == "Gravity Residual":
            im = ax.imshow(values, origin="lower", extent=extent, cmap="RdBu_r",
                           vmin=-residual_max, vmax=residual_max)
        else:
            im = ax.imshow(values, origin="lower", extent=extent, cmap="viridis",
                           vmin=-vmax if float(np.min(values)) < 0 else 0, vmax=vmax)
        ax.set(title=title, xlabel="Easting (m)", ylabel="Northing (m)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Gravity (mGal)")
    fig.suptitle(f"Gravity Forward-Consistency Evaluation — {sample_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# 5. COMPARISONS: ONLY COMPARE MATCHING SINGLE-BODY DATASETS
# ============================================================

def write_baseline_comparison(
    rows: list[dict], split: str, output: Path, prediction_dir: Path,
    experiment: str,
) -> None:
    """Compare E02/E03 with E01 only when saved true density is identical.

    E04 is intentionally excluded: sample IDs are reused across datasets,
    but the underlying models are not the same.
    """
    base_root = EXPERIMENTS["E01"]["output"] / "predictions"
    base_csv = base_root / f"{split}_metrics.csv"
    if not base_csv.is_file():
        print(f"E01 metrics missing; skipping comparison: {base_csv}", flush=True)
        return
    with base_csv.open(newline="", encoding="utf-8") as stream:
        baseline = {r["sample_id"]: r for r in csv.DictReader(stream)}

    metrics = (
        "density_mae", "density_rmse", "density_mae_inside_true_body",
        "body_iou", "body_dice", "body_volume_ratio",
        "top_depth_error_m", "bottom_depth_error_m", "thickness_error_m",
    )
    columns = ["sample_id", "data_alignment_verified"]
    for name in metrics:
        columns.extend((f"e01_{name}", f"{experiment.lower()}_{name}",
                        f"{experiment.lower()}_minus_e01_{name}"))

    comparison_rows = []
    for row in rows:
        sid = str(row["sample_id"])
        if sid not in baseline:
            continue
        base_npz = base_root / split / f"{sid}.npz"
        experiment_npz = prediction_dir / f"{sid}.npz"
        if not base_npz.is_file() or not experiment_npz.is_file():
            continue
        with np.load(base_npz) as base, np.load(experiment_npz) as current:
            aligned = (
                "density" in base.files and "density" in current.files
                and np.array_equal(base["density"], current["density"])
            )
        if not aligned:
            print(f"Skipping {sid}: E01/{experiment} density targets differ.", flush=True)
            continue
        result = {"sample_id": sid, "data_alignment_verified": True}
        for name in metrics:
            if name not in baseline[sid]:
                continue
            old, new = float(baseline[sid][name]), float(row[name])
            result[f"e01_{name}"] = old
            result[f"{experiment.lower()}_{name}"] = new
            result[f"{experiment.lower()}_minus_e01_{name}"] = new - old
        comparison_rows.append(result)

    if not comparison_rows:
        print(f"No aligned E01/{experiment} predictions available; skipping comparison.", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"E01/{experiment} comparison: {output}", flush=True)


def write_e04_e05_comparison(rows: list[dict], split: str,
                             prediction_dir: Path, metrics_dir: Path) -> None:
    """Compare E05 vs E04 only on samples with exactly identical truth.

    This is a valid test-set comparison because E05 is trained and evaluated
    on the E04 multi-body dataset. The E04 predictions must already exist.
    """
    baseline_root = EXPERIMENTS["E04"]["output"] / "predictions"
    baseline_csv = baseline_root / f"{split}_metrics.csv"
    if not baseline_csv.is_file():
        print(f"E04 metrics not found; skipping E04/E05 comparison: {baseline_csv}", flush=True)
        return
    with baseline_csv.open(newline="", encoding="utf-8") as stream:
        baseline = {row["sample_id"]: row for row in csv.DictReader(stream)}

    metric_names = (
        "density_mae", "density_rmse", "density_mae_inside_true_body",
        "body_iou", "body_dice", "body_volume_ratio",
        "top_depth_error_m", "bottom_depth_error_m", "thickness_error_m",
    )
    fields = ["sample_id"]
    for name in metric_names:
        fields += [f"e04_{name}", f"e05_{name}", f"e05_minus_e04_{name}"]

    joined_rows = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in baseline:
            continue
        e04_file = baseline_root / split / f"{sample_id}.npz"
        e05_file = prediction_dir / f"{sample_id}.npz"
        if not e04_file.is_file() or not e05_file.is_file():
            continue
        with np.load(e04_file) as e04_archive, np.load(e05_file) as e05_archive:
            aligned = np.array_equal(e04_archive["density"], e05_archive["density"])
        if not aligned:
            print(f"Skipping {sample_id}: E04 and E05 truths differ.", flush=True)
            continue
        result = {"sample_id": sample_id}
        for metric in metric_names:
            old = float(baseline[sample_id][metric])
            new = float(row[metric])
            result[f"e04_{metric}"] = old
            result[f"e05_{metric}"] = new
            result[f"e05_minus_e04_{metric}"] = new - old
        joined_rows.append(result)

    if not joined_rows:
        print("No aligned E04/E05 saved predictions available; skipping comparison.", flush=True)
        return
    output = metrics_dir / f"E04_vs_E05_{split}_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(joined_rows)
    print(f"E04/E05 aligned comparison saved: {output}", flush=True)


# ============================================================
# 6. PREDICTION AND REPORTING
# ============================================================

def _normalize_sample(sample: str) -> str:
    """Accept '3', '000003', 'sample_000003', or 'sample_000003.npz'."""
    stem = sample.strip().removesuffix(".npz")
    if stem.startswith("sample_"):
        stem = stem[len("sample_"):]
    return f"sample_{int(stem):06d}.npz" if stem.isdigit() else stem + ".npz"


def recreate_density_figures(predictions_dir: Path, figures_dir: Path,
                             samples: list[str] | None, limit: int | None) -> None:
    """Replot already saved predictions without loading any Keras model.

    Use this after changing plotting code; it does not overwrite the NPZ
    prediction arrays, metrics, or training results.
    """
    paths = sorted(predictions_dir.glob("*.npz"))
    if samples is not None:
        wanted = {_normalize_sample(sample) for sample in samples}
        missing = wanted - {p.name for p in paths}
        if missing:
            print(f"Warning: missing saved predictions: {sorted(missing)}", flush=True)
        paths = [p for p in paths if p.name in wanted]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No matching saved predictions in {predictions_dir}")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for index, archive_path in enumerate(paths, 1):
        with np.load(archive_path) as archive:
            density = np.asarray(archive["density"], dtype=np.float32)
            prediction = np.asarray(archive["prediction"], dtype=np.float32)
        if density.shape != VOLUME_SHAPE or prediction.shape != VOLUME_SHAPE:
            raise ValueError(f"Unexpected density shape in {archive_path}")
        dest = figures_dir / f"{archive_path.stem}_density_comparison.png"
        save_comparison_figure(density, prediction, dest, archive_path.stem)
        print(f"Replotted {index}/{len(paths)}: {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="E04")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override preprocessed data; each experiment has its own default.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--gravity-root", type=Path, default=None,
                        help="Override original gravity; defaults depend on the experiment.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Select sample IDs, e.g., --samples 3 5 6.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Recreate density PNGs from existing prediction NPZs; no model inference.")
    parser.add_argument("--visualize-count", type=int, default=None,
                        help="Number of samples to plot; 0 disables figures (default: all).")
    parser.add_argument("--plotly", action="store_true",
                        help="Export Plotly GIFs only (unless --no-gifs is specified).")
    parser.add_argument("--no-gifs", action="store_true",
                        help="With --plotly, export HTML instead of GIFs.")
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly allow writing into a directory with existing results.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.visualize_count is not None and args.visualize_count < 0:
        parser.error("--visualize-count must be >= 0")
    if args.gif_frames < 2:
        parser.error("--gif-frames must be >= 2")

    # Select appropriate paths for *each* experiment. In particular, E04 must
    # NOT accidentally load E01 single-body density or gravity.
    settings = EXPERIMENTS[args.experiment]
    data_root = absolute_path(args.data_root or settings["data"])
    output_root = absolute_path(args.output_root or settings["output"])
    model_path = absolute_path(args.model or settings["training"] / "checkpoints" / "best_unet3d.keras")
    gravity_root = absolute_path(args.gravity_root or settings["gravity"])

    # Plot-only mode reads EXISTING NPZ files and replaces density-section PNGs.
    # It bypasses model loading and all overwrite protection for prediction data.
    if args.plots_only:
        recreate_density_figures(
            output_root / "predictions" / args.split,
            output_root / "figures" / args.split,
            args.samples, args.limit,
        )
        return

    # All experiments except E01 are protected from accidental overwrite; keep E01's legacy
    # behavior unchanged. Custom smoke-test output folders are safe to use.
    if (args.experiment != "E01" and not args.overwrite
            and output_root.exists()
            and any(p.is_file() for p in output_root.rglob("*"))):
        parser.error(f"Output directory is not empty: {output_root}; use --overwrite or --output-root.")

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    files = sample_paths(data_root, args.split, None)
    if args.samples is not None:
        wanted = {_normalize_sample(name) for name in args.samples}
        files = [path for path in files if path.name in wanted]
        if not files:
            raise FileNotFoundError(f"Requested samples not found in {data_root / args.split}")
    if args.limit is not None:
        files = files[:args.limit]
    predictions_dir = output_root / "predictions" / args.split
    figures_dir = output_root / "figures" / args.split
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize_count != 0:
        figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment}", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Preprocessed data: {data_root}", flush=True)
    print(f"Original gravity: {gravity_root}", flush=True)
    print(f"Samples: {len(files)}", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)

    rows: list[dict] = []
    for index, path in enumerate(files, start=1):
        input_volume, target = load_sample(path)
        prediction = model(input_volume[None, ...], training=False).numpy()[0, ..., 0]
        truth = target[..., 0]
        input_volume = input_volume[..., 0]
        if not np.isfinite(prediction).all():
            raise ValueError(f"{path.name}: prediction contains NaN or infinity")
        if prediction.shape != VOLUME_SHAPE:
            raise ValueError(f"{path.name}: predicted shape {prediction.shape}, expected {VOLUME_SHAPE}")
        if prediction.min() < 0 or prediction.max() > 1:
            raise ValueError(f"{path.name}: prediction outside sigmoid [0, 1] range")

        with np.load(path) as archive:
            migration = np.asarray(archive["migration"], dtype=np.float32)
            migration_scale = float(archive["migration_scale"])

        metrics = calculate_metrics(truth, prediction)
        rows.append({"sample_id": path.stem, **metrics})
        np.savez_compressed(
            predictions_dir / path.name,
            prediction=prediction.astype(np.float32),
            density=truth.astype(np.float32),
            migration=migration,
            migration_input=input_volume.astype(np.float32),
            migration_scale=np.float64(migration_scale),
        )

        # Plots are optional. With --visualize-count 0 no gravity loading or
        # forward calculation is needed, which makes smoke tests fast.
        if args.visualize_count is None or index <= args.visualize_count:
            save_comparison_figure(
                truth, prediction,
                figures_dir / f"{path.stem}_density_comparison.png", path.stem,
            )

            observed_path = gravity_root / args.split / path.name
            if not observed_path.is_file():
                raise FileNotFoundError(
                    f"Original gravity file missing: {observed_path}\n"
                    "Use --gravity-root to point to the correct forward_models directory."
                )
            with np.load(observed_path) as archive:
                observed = np.asarray(archive["gravity"], dtype=np.float32)
            if observed.shape != GRAVITY_SHAPE:
                raise ValueError(f"{observed_path}: gravity shape {observed.shape}; expected {GRAVITY_SHAPE}")

            # Forward model the network's predicted density and compare to
            # the observed gravity. This uses the existing TF forward proxy.
            recovered = gravity_forward_tf(
                tf.convert_to_tensor(prediction[None, ..., None], dtype=tf.float32)
            ).numpy()[0, ..., 0]
            if recovered.shape != GRAVITY_SHAPE or not np.isfinite(recovered).all():
                raise ValueError(f"{path.stem}: invalid recovered gravity {recovered.shape}")
            save_gravity_comparison(
                observed, recovered,
                figures_dir / f"{path.stem}_gravity_comparison.png", path.stem,
            )

            # Interactive Plotly figures and GIFs are opt-in.
            if args.plotly:
                try:
                    from visualization.plotly_3d import plot_sample_3d
                    plot_sample_3d(
                        truth, observed, figures_dir / f"{path.stem}_3d.{'html' if args.no_gifs else 'gif'}",
                        reconstruction=prediction,
                        make_gif=not args.no_gifs,
                        gif_frames=args.gif_frames,
                        gif_duration_seconds=8.0,
                    )
                except (ImportError, RuntimeError) as error:
                    print(f"Optional Plotly export skipped: {error}", flush=True)

        print(
            f"Predicted {index}/{len(files)}: {path.stem}; "
            f"MAE={metrics['density_mae']:.5g}; IoU={metrics['body_iou']:.4f}; "
            f"empty={metrics['empty_prediction']}", flush=True,
        )

    metrics_dir = output_root / "predictions"
    metrics_csv = metrics_dir / f"{args.split}_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = [key for key in rows[0] if key not in ("sample_id", "empty_prediction")]
    summary = {
        "experiment": args.experiment,
        "split": args.split,
        "sample_count": len(rows),
        "empty_prediction_count": sum(bool(row["empty_prediction"]) for row in rows),
        "density_threshold_gcm3": DENSITY_THRESHOLD,
        "empty_prediction_depth_penalty_m": VOLUME_SHAPE[0] * CELL_SIZE_M,
        "data_root": str(data_root),
        "model": str(model_path),
        "metrics": {
            name: {
                "mean": float(np.mean([float(r[name]) for r in rows])),
                "std": float(np.std([float(r[name]) for r in rows])),
            }
            for name in numeric_fields
        },
    }
    summary_path = metrics_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Comparisons require identical test targets; E04 multi-body samples cannot
    # be matched to E01 single-body samples by their filenames alone.
    if args.experiment in {"E02_balanced_huber", "E03"}:
        short_name = "E02" if args.experiment == "E02_balanced_huber" else "E03"
        write_baseline_comparison(
            rows, args.split,
            metrics_dir / f"E01_vs_{short_name}_{args.split}_metrics.csv",
            predictions_dir, short_name,
        )
    elif args.experiment == "E05":
        # E05 and E04 share the same held-out dataset. Comparison still
        # verifies exact target-array alignment for every sample.
        write_e04_e05_comparison(rows, args.split, predictions_dir, metrics_dir)
    elif args.experiment in {"E04", "E04B"}:
        print(f"{args.experiment}: no cross-dataset comparison by sample ID.", flush=True)

    print(json.dumps(summary, indent=2))
    print(f"Metrics: {metrics_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
