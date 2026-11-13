"""Prepare one Thunderbird Gzz survey tile for existing E08D inference.

Run from the root of ml-enhanced-gravity-migration.
This does NOT create density labels or retrain the E08D loss.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from src.physics.gravity_forward import OBS_X, OBS_Y
from src.physics.gravity_migration import calculate_migration, normalize_migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ml/training/Thunderbird/data/test/field/sample_000001.npz"))
    parser.add_argument("--channel", type=int, default=7,
                        help="Gzz channel in the repository's legacy FWD3D convention (verify source metadata)")
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--max-nearest-distance", type=float, default=200.0,
                        help="Reject if any interpolated station is farther than this many metres from data")
    args = parser.parse_args()

    values = np.loadtxt(args.input)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("Expected finite, five-column text data: x y z component value")

    rows = values[values[:, 3] == args.channel]
    if not len(rows):
        raise ValueError(f"Missing component {args.channel}")
    if np.any(rows[:, 2] != rows[0, 2]):
        raise ValueError("Variable observation z; the repo migration assumes a flat observation plane")

    # The repo's observation coordinates are -85..715 m in both directions,
    # while its model occupies x/y=0..640 m. Translate, do not rescale.
    center_x = float(np.median(rows[:, 0])) if args.center_x is None else args.center_x
    center_y = float(np.median(rows[:, 1])) if args.center_y is None else args.center_y
    origin_x, origin_y = center_x - 315.0, center_y - 315.0
    grid_x, grid_y = np.meshgrid(origin_x + OBS_X, origin_y + OBS_Y)
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Crop for speed, retaining a margin for triangulation.
    margin = 150.0
    subset = (
        (rows[:, 0] >= grid_x.min() - margin) &
        (rows[:, 0] <= grid_x.max() + margin) &
        (rows[:, 1] >= grid_y.min() - margin) &
        (rows[:, 1] <= grid_y.max() + margin)
    )
    local = rows[subset]
    if len(local) < 10:
        raise ValueError("Insufficient observations near selected tile")

    distances, _ = cKDTree(local[:, :2]).query(targets)
    if distances.max() > args.max_nearest_distance:
        raise ValueError(
            f"Insufficient spatial coverage: maximum nearest-observation "
            f"distance = {distances.max():.1f} m. Try another tile center."
        )

    interpolated = griddata(local[:, :2], local[:, 4], targets, method="linear")
    missing = ~np.isfinite(interpolated)
    if missing.mean() > 0.02:
        raise ValueError(f"{missing.mean():.1%} of tile lies outside the station convex hull; choose another center")
    if missing.any():
        interpolated[missing] = griddata(local[:, :2], local[:, 4],
                                         targets[missing], method="nearest")
    gravity = interpolated.reshape(81, 81).astype(np.float32)

    migration = calculate_migration(gravity, component="gzz", restrict=False)
    migration_input, migration_scale = normalize_migration(migration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gravity=gravity,
        migration=migration.astype(np.float32),
        migration_input=migration_input,
        migration_scale=np.float64(migration_scale),
        observation_component=np.str_("GZZ"),
        observation_unit=np.str_("E"),  # Assumed; confirm original survey metadata
        source_channel=np.int32(args.channel),
        survey_origin_xy=np.array([origin_x, origin_y]),
        survey_center_xy=np.array([center_x, center_y]),
        nearest_station_max_m=np.float64(distances.max()),
    )
    print(f"Wrote {args.output}")
    print(f"Channel {args.channel}: {len(rows)} stations; tile uses {len(local)} nearby stations")
    print(f"Tile center: ({center_x:.2f}, {center_y:.2f}) m")
    print(f"Max nearest-station distance: {distances.max():.1f} m")
    print(f"Nearest-filled fraction (outside convex hull): {missing.mean():.2%}")
    print(f"Gzz range: [{gravity.min():.3f}, {gravity.max():.3f}] assumed E")
    print(f"Migration: shape {migration_input.shape}, scale {migration_scale:.5g}")


if __name__ == "__main__":
    main()
