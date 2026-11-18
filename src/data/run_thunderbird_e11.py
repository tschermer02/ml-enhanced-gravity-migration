"""Thunderbird E11: prepare field Gzz, migrate, and infer 3-D density.

Designed for the E11 signed 3-D U-Net trained on a 24x64x64 E11 grid:
  Model: 8 km x 8 km x 1 km (125, 125, 41.6667 m cells)
  Observations: 81x81 Gzz in Eotvos (E), flat acquisition at nominal height.

This is field inference using a synthetic-trained model, NOT field training or
proof of density accuracy. Interpolation does not add survey resolution.

From the repository root:

  python -m src.prediction.run_thunderbird_e11 \\
    --experiment E11E \\
    --input src/data/Thunderbird_gravity_data/Tbird_gravity.dat \\
    --height 100 --input-unit eotvos

Default output: outputs/ml/field/Thunderbird_E11/E11E/

The 81x81 E11 receiver footprint is ~10 km wide and extends beyond the 8 km
density model. Stations outside the measured coverage are nearest-extrapolated
and EXPLICITLY flagged. Use --strict-coverage to refuse extrapolation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from src.physics.gravity_forward import (
    E11_DX_M, E11_DY_M, E11_DZ_M, SHAPE, OBS_SHAPE,
    observation_coordinates, calculate_gzz,
)
from src.physics.gravity_migration import (
    calculate_migration, normalize_migration,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {"E11BASE", "E11A", "E11B", "E11C", "E11D", "E11E"}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_field(path: Path, channel: int, input_unit: str):
    """Read five-column x,y,z,component,value survey and convert Gzz to E."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
        raise ValueError("Expected five finite columns: x y z component value")
    field = data[data[:, 3] == channel]
    if len(field) < 4:
        raise ValueError(f"Found fewer than four observations for channel {channel}")
    # Legacy field-processing code assumes one plane. Height is a declared
    # modeling assumption rather than inferred from unknown source-z datum.
    if not np.allclose(field[:, 2], field[0, 2], rtol=0, atol=1e-5):
        raise ValueError("The survey has varying z. A single flat observation height "
                         "is invalid; preprocess/topography-correct data first.")
    xy = field[:, :2]
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("Duplicate XY stations; resolve/average duplicates first.")
    values = field[:, 4].copy()
    if input_unit == "si":
        values *= 1e9  # 1/s^2 -> Eotvos
    elif input_unit != "eotvos":
        raise ValueError("input_unit must be 'eotvos' or 'si'")
    return xy, values, float(field[0, 2])


def prepare_field(xy, values, *, center_x, center_y, height_m,
                  max_distance_m, strict_coverage):
    """Interpolate on the exact E11 81x81 observation coordinates.

    A model tile begins at (center_x-4000, center_y-4000). Its receiver grid
    intentionally extends beyond the 8 km model: ~10 km total footprint.
    """
    origin_x = float(center_x) - 4000.0
    origin_y = float(center_y) - 4000.0
    obs_x, obs_y = observation_coordinates(E11_DX_M, E11_DY_M)
    gx, gy = np.meshgrid(origin_x + obs_x, origin_y + obs_y)
    points = np.column_stack((gx.ravel(), gy.ravel()))
    distances = cKDTree(xy).query(points)[0]

    linear = LinearNDInterpolator(xy, values, fill_value=np.nan)
    interpolated = np.asarray(linear(points), dtype=np.float64).reshape(-1)
    outside_hull = ~np.isfinite(interpolated)
    far_from_station = distances > float(max_distance_m)
    invalid = outside_hull | far_from_station

    if strict_coverage and invalid.any():
        raise ValueError(
            f"{np.mean(invalid):.1%} of E11 receiver sites are outside the survey "
            f"convex hull or > {max_distance_m:g} m from a measured station; "
            "choose another center or run without --strict-coverage and inspect masks."
        )
    if outside_hull.any():
        nearest = NearestNDInterpolator(xy, values)
        interpolated[outside_hull] = nearest(points[outside_hull])

    gravity = interpolated.reshape(OBS_SHAPE).astype(np.float32)
    if not np.isfinite(gravity).all():
        raise ValueError("Nonfinite interpolated Gzz")

    migration = calculate_migration(
        gravity, component="gzz", restrict=False,
        dx_m=E11_DX_M, dy_m=E11_DY_M, dz_m=E11_DZ_M,
        observation_height_m=float(height_m),
    )
    migration_input, migration_scale = normalize_migration(migration)
    if migration_input.shape != SHAPE or not np.isfinite(migration_input).all():
        raise ValueError("Invalid E11 migrated volume")
    return {
        "gravity": gravity,
        "migration": migration.astype(np.float32),
        "migration_input": migration_input.astype(np.float32),
        "migration_scale": np.float64(migration_scale),
        "observation_height_m": np.float64(height_m),
        "dx_m": np.float64(E11_DX_M),
        "dy_m": np.float64(E11_DY_M),
        "dz_m": np.float64(E11_DZ_M),
        "observation_component": np.str_("GZZ"),
        "observation_unit": np.str_("E"),
        "survey_origin_xy": np.array([origin_x, origin_y], np.float64),
        "survey_center_xy": np.array([center_x, center_y], np.float64),
        "x_cell_centers_m": origin_x + (np.arange(64)+0.5)*E11_DX_M,
        "y_cell_centers_m": origin_y + (np.arange(64)+0.5)*E11_DY_M,
        "z_cell_centers_m": (np.arange(24)+0.5)*E11_DZ_M,
        "receiver_x_m": origin_x + obs_x,
        "receiver_y_m": origin_y + obs_y,
        "receiver_outside_hull_mask": outside_hull.reshape(OBS_SHAPE),
        "receiver_far_from_station_mask": far_from_station.reshape(OBS_SHAPE),
        "nearest_station_distance_m": distances.reshape(OBS_SHAPE).astype(np.float32),
    }


def predict(model_path: Path, migration_input):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    expected = (None, 24, 64, 64, 1)
    if tuple(model.input_shape) != expected or tuple(model.output_shape) != expected:
        raise ValueError(f"Expected model input/output {expected}; found "
                         f"{model.input_shape} and {model.output_shape}")
    result = model(migration_input[None, ..., None], training=False)
    prediction = np.asarray(result.numpy(), dtype=np.float32)
    if prediction.shape != (1, 24, 64, 64, 1) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predicted density shape {prediction.shape}")
    return prediction[0, ..., 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--input", type=Path,
                        default=Path("src/data/Thunderbird_gravity_data/Tbird_gravity.dat"))
    parser.add_argument("--model", type=Path, default=None,
                        help="Override E11 checkpoint path")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--channel", type=int, default=7,
                        help="Legacy Gzz channel. Verify the survey column convention.")
    parser.add_argument("--input-unit", choices=("eotvos", "si"), required=True,
                        help="Declare measured Gzz unit; SI means s^-2.")
    parser.add_argument("--height", type=float, default=100.0,
                        help="ASSUMED flat receiver height above model top, m")
    parser.add_argument("--center-x", type=float)
    parser.add_argument("--center-y", type=float)
    parser.add_argument("--max-distance", type=float, default=350.0,
                        help="Flag observation nodes farther than this from a station (m)")
    parser.add_argument("--strict-coverage", action="store_true",
                        help="Refuse extrapolated/poorly sampled receiver nodes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.height) or args.height < 0:
        parser.error("--height must be nonnegative and finite")
    if not np.isfinite(args.max_distance) or args.max_distance <= 0:
        parser.error("--max-distance must be positive and finite")

    source = _resolve(args.input)
    checkpoint = _resolve(args.model or
                          f"outputs/ml/training/{args.experiment}/checkpoints/best_unet3d.keras")
    out = _resolve(args.output_root or
                   f"outputs/ml/field/Thunderbird_E11/{args.experiment}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Avoid silently passing an old 10m-grid checkpoint just because tensor
    # dimensions are the same.
    config_path = checkpoint.parent.parent / "logs" / "training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing E11 training metadata {config_path}; cannot verify grid."
        )
    training = json.loads(config_path.read_text(encoding="utf-8"))
    if training.get("experiment") != args.experiment or training.get("grid") != "e11":
        raise ValueError("Checkpoint training metadata does not match E11 experiment")
    for key, required in (("dx_m", E11_DX_M), ("dy_m", E11_DY_M),
                          ("dz_m", E11_DZ_M)):
        if not np.isclose(float(training.get(key, -1)), required):
            raise ValueError(f"Checkpoint {key} does not match E11")

    target_pre = out / "Thunderbird_e11_preprocessed.npz"
    target_pred = out / "Thunderbird_e11_prediction.npz"
    if not args.overwrite and (target_pre.exists() or target_pred.exists()):
        raise FileExistsError("Thunderbird E11 output exists; pass --overwrite")

    xy, gzz, source_z = load_field(source, args.channel, args.input_unit)
    cx = float(args.center_x if args.center_x is not None else np.median(xy[:, 0]))
    cy = float(args.center_y if args.center_y is not None else np.median(xy[:, 1]))
    if not np.isfinite([cx, cy]).all():
        raise ValueError("Invalid center coordinates")

    prepared = prepare_field(
        xy, gzz, center_x=cx, center_y=cy, height_m=args.height,
        max_distance_m=args.max_distance,
        strict_coverage=args.strict_coverage,
    )
    outside = prepared["receiver_outside_hull_mask"]
    far = prepared["receiver_far_from_station_mask"]
    print(f"Thunderbird stations: {len(xy)}")
    print(f"Observation coverage: outside hull {outside.mean():.1%}; "
          f"> {args.max_distance:g} m from stations {far.mean():.1%}", flush=True)
    if outside.any() or far.any():
        print("WARNING: field observations contain interpolated/extrapolated or "
              "undersampled receiver sites; results are not fully observation-constrained.")

    density = predict(checkpoint, prepared["migration_input"])
    modeled_gzz = calculate_gzz(
        density, dx_m=E11_DX_M, dy_m=E11_DY_M, dz_m=E11_DZ_M,
        observation_height_m=args.height,
    ).astype(np.float32)
    residual = prepared["gravity"] - modeled_gzz
    # Report values only over near-station, inside-hull receiver locations.
    valid = ~(outside | far)
    rmse = (float(np.sqrt(np.mean(residual[valid]**2)))
            if np.any(valid) else None)
    if rmse is None:
        print("WARNING: No covered receiver nodes, so covered-only RMSE is undefined.")

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_pre, **prepared)
    np.savez_compressed(
        target_pred, **prepared,
        prediction=density, modeled_gravity=modeled_gzz,
        gravity_residual=residual.astype(np.float32),
    )
    meta = {
        "experiment": args.experiment,
        "checkpoint": str(checkpoint),
        "input": str(source),
        "channel": args.channel,
        "input_unit": args.input_unit,
        "field_z_coordinate": source_z,
        "assumed_height_above_model_m": args.height,
        "center_xy_m": [cx, cy],
        "model_extent_m": [8000.0, 8000.0, 1000.0],
        "grid_shape": list(SHAPE),
        "station_count": int(len(xy)),
        "fraction_receiver_outside_hull": float(outside.mean()),
        "fraction_receiver_far_from_station": float(far.mean()),
        "fraction_receiver_covered": float(valid.mean()),
        "covered_receiver_rmse_e": rmse,
        "limitations": [
            "Survey Gzz channel and units require independent source verification.",
            "Receiver height is assumed flat relative to the model top.",
            "Interpolation/extrapolation does not add measurement resolution.",
            "Synthetic-trained predictions are not validated ground truth.",
            "Gzz uses a cell-centered point-mass forward approximation.",
        ],
    }
    (out / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved preprocessing: {target_pre}")
    print(f"Saved prediction: {target_pred}")
    print(f"Covered-site Gzz RMSE: {rmse if rmse is not None else 'undefined'} E")


if __name__ == "__main__":
    main()
