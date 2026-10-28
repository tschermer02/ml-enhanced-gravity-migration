"""Metrics and aggregation for signed or unsigned 3D density predictions."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def calculate_metrics(truth, prediction, threshold=0.1, cell_size_m=10.0):
    """Score density and recovered support; occupancy uses absolute contrast."""
    true = np.asarray(truth, dtype=np.float64).squeeze()
    pred = np.asarray(prediction, dtype=np.float64).squeeze()
    if true.shape != pred.shape or true.ndim != 3:
        raise ValueError("truth and prediction must be 3D arrays with matching shapes")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("truth and prediction must be finite")
    if threshold <= 0 or cell_size_m <= 0:
        raise ValueError("threshold and cell_size_m must be positive")

    t = np.abs(true) >= threshold
    p = np.abs(pred) >= threshold
    nt, npred = int(t.sum()), int(p.sum())
    intersection = int((t & p).sum())
    union = int((t | p).sum())
    err = pred - true

    def extent(mask):
        indices = np.flatnonzero(mask.any(axis=(1, 2)))
        if len(indices) == 0:
            return None
        return (indices[0] * cell_size_m, (indices[-1] + 1) * cell_size_m)

    td, pd = extent(t), extent(p)
    if td is None or pd is None:
        top_error = bottom_error = thickness_error = np.nan
    else:
        top_error = abs(pd[0] - td[0])
        bottom_error = abs(pd[1] - td[1])
        thickness_error = abs((pd[1] - pd[0]) - (td[1] - td[0]))
    signed_intersection = int((t & p & (np.sign(true) == np.sign(pred))).sum())
    return {
        "density_mae": float(np.mean(np.abs(err))),
        "density_rmse": float(np.sqrt(np.mean(err**2))),
        "density_mae_inside_true_body": float(np.mean(np.abs(err[t]))) if nt else np.nan,
        "body_iou": intersection / union if union else 1.0,
        "body_dice": 2 * intersection / (nt + npred) if nt + npred else 1.0,
        "signed_body_iou": signed_intersection / union if union else 1.0,
        "true_body_sign_accuracy": float(np.mean(np.sign(true[t]) == np.sign(pred[t]))) if nt else np.nan,
        "body_volume_ratio": npred / nt if nt else np.nan,
        "top_depth_error_m": float(top_error),
        "bottom_depth_error_m": float(bottom_error),
        "thickness_error_m": float(thickness_error),
        "empty_prediction": not bool(npred),
    }


def summarize(rows):
    """Compute sample count and mean/std for numeric metric columns."""
    if not rows:
        raise ValueError("Cannot summarize an empty set of metrics")
    excluded = {"sample_id", "split", "geometry", "noise_percent", "empty_prediction"}
    names = [k for k in rows[0] if k not in excluded and isinstance(rows[0][k], (int, float, np.number)) and not isinstance(rows[0][k], (bool, np.bool_))]
    stats = {}
    for key in names:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        valid = values[np.isfinite(values)]
        stats[key] = {"mean": float(valid.mean()) if len(valid) else None,
                      "std": float(valid.std()) if len(valid) else None,
                      "valid_count": int(len(valid))}
    return {"sample_count": len(rows),
            "empty_prediction_count": sum(bool(row["empty_prediction"]) for row in rows),
            "metrics": stats}


def grouped_summary(rows):
    """Group on geometry and noise level when present, without named cases."""
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row.get("geometry", "unknown")), float(row.get("noise_percent", 0)))].append(row)
    return {geometry: {f"noise_{noise:g}": summarize(group)
                       for (g, noise), group in groups.items() if g == geometry}
            for geometry, _ in groups}
