"""Evaluate a saved 3D density model, with E11 physical-grid metadata.

Legacy usage:
    python -m src.evaluation.evaluate --name E08D --data-root outputs/ml/training/D06C/data

E11 usage:
    python -m src.evaluation.evaluate --name E11A --split test

Depth metrics use dz_m, not dx_m, because z is the first tensor axis.
Existing metric names and legacy CSV/JSON outputs are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import calculate_metrics, grouped_summary, summarize

ROOT = Path(__file__).resolve().parents[2]

E11_DATASETS = {
    "E11BASE": "D11_BASE",
    "E11A": "D11_INTRUSIVE",
    "E11B": "D11_HEIGHT",
    "E11C": "D11_BASE",
    "E11D": "D11_COMBINED",
    "E11E": "D11_COMBINED",
}

METADATA_FIELDS = (
    "sample_id", "split", "geometry", "noise_percent",
    "observation_height_m", "dx_m", "dy_m", "dz_m",
)


def _scalar(archive, key, fallback):
    if key not in archive:
        return float(fallback)
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar, got shape {value.shape}")
    result = float(value.item())
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(rows):
    """Exclude dataset metadata so it is never reported as a performance metric."""
    return [
        {k: v for k, v in row.items()
         if k not in METADATA_FIELDS}
        for row in rows
    ]


def _summary(rows):
    return summarize(_metric_rows(rows))


def _by_height(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["observation_height_m"])].append(row)
    return {
        f"height_{height:g}_m": _summary(grouped[height])
        for height in sorted(grouped)
    }


def _by_geometry_noise_height(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (str(row["geometry"]),
               float(row["noise_percent"]),
               float(row["observation_height_m"]))
        groups[key].append(row)

    nested = {}
    for (geom, noise, height), group in sorted(groups.items()):
        nested.setdefault(geom, {}).setdefault(
            f"noise_{noise:g}", {}
        )[f"height_{height:g}_m"] = _summary(group)
    return nested


def _data_config(data_root):
    path = data_root / "config.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    return {}


def evaluate(data_root, model_path, output_root, split="test", *, threshold=0.1,
             cell_size_m=10.0, limit=None, overwrite=False, e11=False):
    """Evaluate predictions, reusing existing saved prediction NPZs.

    `cell_size_m` remains the legacy fallback for vertical spacing.
    E11 samples must provide dx_m, dy_m, dz_m and observation_height_m.
    """
    import tensorflow as tf  # Lazy import retains original import behavior.

    data_root = Path(data_root)
    model_path = Path(model_path)
    output_root = Path(output_root)

    if threshold <= 0 or cell_size_m <= 0:
        raise ValueError("threshold and fallback cell size must be positive")

    files = sorted((data_root / split).rglob("*.npz"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No {split} samples in {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    marker = output_root / f"{split}_summary.json"
    if marker.exists() and not overwrite:
        raise FileExistsError(f"Evaluation exists at {marker}; use --overwrite")

    config = _data_config(data_root)
    e11 = bool(e11 or config.get("grid") == "e11")
    model = None
    rows = []

    for index, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as sample:
            x = np.asarray(sample["migration_input"], dtype=np.float32)
            truth = np.asarray(sample["density"], dtype=np.float32)
            geometry = (str(sample["geometry"].item())
                        if "geometry" in sample else "unknown")
            noise = _scalar(sample, "noise_percent", 0.0)

            if e11:
                missing = {"dx_m", "dy_m", "dz_m", "observation_height_m"} - set(sample.files)
                if missing:
                    raise KeyError(f"{path}: missing E11 metadata: {sorted(missing)}")

            dx = _scalar(sample, "dx_m", cell_size_m)
            dy = _scalar(sample, "dy_m", cell_size_m)
            dz = _scalar(sample, "dz_m", cell_size_m)
            height = _scalar(sample, "observation_height_m", 0.0)

        if min(dx, dy, dz) <= 0 or height < 0:
            raise ValueError(f"{path}: invalid cell dimensions or observation height")
        if x.shape != (24, 64, 64) or truth.shape != x.shape:
            raise ValueError(f"{path}: unexpected model shape {x.shape}, {truth.shape}")
        if not np.isfinite(x).all() or not np.isfinite(truth).all():
            raise ValueError(f"{path}: nonfinite input/target")

        if e11 and not (np.isclose(dx, 125.0)
                        and np.isclose(dy, 125.0)
                        and np.isclose(dz, 1000.0 / 24.0)):
            raise ValueError(f"{path}: expected E11 grid 125 x 125 x 41.667 m")

        relative = path.relative_to(data_root / split)
        pred_path = output_root.joinpath(
            *relative.parts[:-1], "predictions", split, path.name
        )
        if pred_path.exists():
            with np.load(pred_path, allow_pickle=False) as saved:
                pred = np.asarray(saved["prediction"], dtype=np.float32).squeeze()
                if e11:
                    for key, expected in (
                        ("dx_m", dx), ("dy_m", dy), ("dz_m", dz),
                        ("observation_height_m", height)
                    ):
                        if key not in saved or not np.isclose(
                            float(np.asarray(saved[key]).item()), expected
                        ):
                            raise ValueError(
                                f"{pred_path}: saved {key} mismatches source sample"
                            )
        else:
            if model is None:
                model = tf.keras.models.load_model(model_path, compile=False)
            pred = np.asarray(
                model(x[None, ..., None], training=False).numpy(),
                dtype=np.float32
            ).squeeze()
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with np.load(path, allow_pickle=False) as sample:
                payload = {key: sample[key] for key in sample.files}
            np.savez_compressed(
                pred_path, **payload, prediction=pred.astype(np.float32)
            )

        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"{pred_path}: invalid prediction")

        # calculate_metrics uses its cell_size_m argument ONLY for z/depth.
        # Passing dz here fixes physical depth errors for anisotropic E11 cells.
        metrics = calculate_metrics(truth, pred, threshold, dz)
        rows.append({
            "sample_id": path.stem,
            "split": split,
            "geometry": geometry,
            "noise_percent": noise,
            **metrics,
            "observation_height_m": height,
            "dx_m": dx,
            "dy_m": dy,
            "dz_m": dz,
        })

        if index % 25 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_rows(output_root / f"{split}_metrics.csv", rows)

    summary = {
        "split": split,
        "overall": _summary(rows),
        "by_geometry_and_noise": grouped_summary(_metric_rows(rows)),
        "by_observation_height": _by_height(rows),
        "by_geometry_noise_and_height": _by_geometry_noise_height(rows),
        "model_path": str(model_path),
        "data_root": str(data_root),
        "threshold_gcm3": threshold,
        "cell_size_m": cell_size_m,
        "vertical_spacing_source": "sample dz_m if available; otherwise --cell-size",
        "grid": config.get("grid", "legacy"),
    }
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for geom in sorted({row["geometry"] for row in rows}):
        group = [row for row in rows if row["geometry"] == geom]
        folder = output_root / geom
        folder.mkdir(parents=True, exist_ok=True)
        _write_rows(folder / f"{split}_metrics.csv", group)
        (folder / f"{split}_summary.json").write_text(
            json.dumps({
                "geometry": geom,
                "overall": _summary(group),
                "by_geometry_and_noise": grouped_summary(_metric_rows(group)),
                "by_observation_height": _by_height(group),
                "by_geometry_noise_and_height": _by_geometry_noise_height(group),
            }, indent=2),
            encoding="utf-8",
        )

    print(f"Finished evaluating {split} dataset: {output_root}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="test")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--cell-size", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    training = ROOT / "outputs" / "ml" / "training" / args.name
    dataset_name = E11_DATASETS.get(args.name, args.name)
    default_data_root = (ROOT / "outputs" / "ml" / "training"
                         / dataset_name / "data")

    evaluate(
        args.data_root or (default_data_root if args.name in E11_DATASETS
                           else training / "data"),
        args.model or training / "checkpoints" / "best_unet3d.keras",
        args.output_root or ROOT / "outputs" / "ml" / "predict" / args.name,
        args.split,
        threshold=args.threshold,
        cell_size_m=args.cell_size,
        limit=args.limit,
        overwrite=args.overwrite,
        e11=args.name in E11_DATASETS,
    )


if __name__ == "__main__":
    main()
