"""Load migration-to-density samples for TensorFlow training.

Modes:
  density       E08D-compatible supervised (migration, density).
  physics       Legacy supervised + observed (possibly noisy) gravity target.
  physics_clean E11 physics training: density, measured/clean gravity, receiver
                height, and anisotropic grid dimensions. Requires E11 metadata.

The input is always the stored, per-sample normalized migration_input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import numpy as np

# TensorFlow is imported only when constructing a tf.data.Dataset.
VOLUME_SHAPE = (24, 64, 64)
GRAVITY_SHAPE = (81, 81)
SPLITS = ("train", "validation", "test")
TargetMode = Literal["density", "physics", "physics_clean"]


def sample_paths(data_root: str | Path, split: str,
                 limit: int | None = None) -> list[Path]:
    """Return sorted sample files for a single split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    root = Path(data_root)
    paths = sorted((root / split).rglob("*.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .npz samples in {root / split}")
    return paths


def _checked_array(archive, key: str, shape: tuple[int, ...], path: Path) -> np.ndarray:
    if key not in archive:
        raise KeyError(f"{path}: missing {key!r} (available: {archive.files})")
    value = np.asarray(archive[key], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{path}: {key} shape {value.shape}, expected {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: {key} has NaN or infinity")
    return value


def _checked_scalar(archive, key: str, path: Path, *, positive: bool = False) -> np.float32:
    """Require a finite, scalar NPZ metadata value."""
    if key not in archive:
        raise KeyError(f"{path}: missing E11 metadata {key!r}")
    value = np.asarray(archive[key], dtype=np.float64)
    if value.shape != () or not np.isfinite(value.item()):
        raise ValueError(f"{path}: {key} must be a finite scalar")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"{path}: {key} must be positive")
    if key == "observation_height_m" and number < 0:
        raise ValueError(f"{path}: observation_height_m must be nonnegative")
    return np.float32(number)


def load_sample(path: str | Path, *, targets: TargetMode = "density",
                gravity_root: str | Path | None = None,
                max_abs_density: float | None = 1.0):
    """Return (migration_input, targets) with trailing channel dimensions.

    targets='density' returns the signed density target (legacy default).
    targets='physics' returns {'density', 'gravity'} using measured gravity,
    with the existing separate-gravity-folder fallback for older experiments.
    targets='physics_clean' returns {'density', 'gravity', 'gravity_clean',
    'observation_height_m', 'dx_m', 'dy_m', 'dz_m'}. It *requires* new unified
    E11 samples, and checks that the gravity component is Gzz.

    Scalar metadata in physics_clean is float32 (no trailing channel dimension).
    max_abs_density=None disables the density-range check.
    """
    path = Path(path)
    if targets not in ("density", "physics", "physics_clean"):
        raise ValueError("targets must be 'density', 'physics', or 'physics_clean'")

    with np.load(path, allow_pickle=False) as archive:
        x = _checked_array(archive, "migration_input", VOLUME_SHAPE, path)
        density = _checked_array(archive, "density", VOLUME_SHAPE, path)

        gravity = None
        clean = None
        metadata = None
        if targets in ("physics", "physics_clean") and "gravity" in archive:
            gravity = _checked_array(archive, "gravity", GRAVITY_SHAPE, path)

        if targets == "physics_clean":
            clean = _checked_array(archive, "gravity_clean", GRAVITY_SHAPE, path)
            component = str(archive["observation_component"].item()).lower() if "observation_component" in archive else None
            if component != "gzz":
                raise ValueError(f"{path}: E11 physics_clean requires observation_component='GZZ' (found {component!r})")
            metadata = {
                "observation_height_m": _checked_scalar(archive, "observation_height_m", path),
                "dx_m": _checked_scalar(archive, "dx_m", path, positive=True),
                "dy_m": _checked_scalar(archive, "dy_m", path, positive=True),
                "dz_m": _checked_scalar(archive, "dz_m", path, positive=True),
            }
            if gravity is None:
                raise KeyError(f"{path}: E11 physics_clean requires measured 'gravity' in the sample NPZ")

    if max_abs_density is not None:
        if max_abs_density <= 0:
            raise ValueError("max_abs_density must be positive or None")
        if np.max(np.abs(density)) > max_abs_density + 1e-6:
            raise ValueError(f"{path}: |density| exceeds {max_abs_density} g/cm^3")

    x = x[..., None]
    density = density[..., None]
    if targets == "density":
        return x, density

    if targets == "physics_clean":
        return x, {
            "density": density,
            "gravity": gravity[..., None],
            "gravity_clean": clean[..., None],
            **metadata,
        }

    if gravity is None:
        if gravity_root is None:
            raise KeyError(f"{path}: no gravity field; provide gravity_root for legacy data")
        gravity_path = Path(gravity_root) / path.parent.name / path.name
        if not gravity_path.exists():
            raise FileNotFoundError(gravity_path)
        with np.load(gravity_path, allow_pickle=False) as archive:
            gravity = _checked_array(archive, "gravity", GRAVITY_SHAPE, gravity_path)

    return x, {"density": density, "gravity": gravity[..., None]}


def make_dataset(data_root: str | Path, split: str, batch_size: int = 2,
                 *, limit: int | None = None, seed: int = 42,
                 targets: TargetMode = "density",
                 gravity_root: str | Path | None = None,
                 max_abs_density: float | None = 1.0):
    """Build a lazy, shuffled (training only), batched tf.data.Dataset."""
    import tensorflow as tf

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if targets not in ("density", "physics", "physics_clean"):
        raise ValueError("targets must be 'density', 'physics', or 'physics_clean'")
    paths = sample_paths(data_root, split, limit)
    volume_spec = tf.TensorSpec((*VOLUME_SHAPE, 1), tf.float32)
    gravity_spec = tf.TensorSpec((*GRAVITY_SHAPE, 1), tf.float32)

    if targets == "density":
        target_spec = volume_spec
    elif targets == "physics":
        target_spec = {"density": volume_spec, "gravity": gravity_spec}
    else:
        scalar_spec = tf.TensorSpec((), tf.float32)
        target_spec = {
            "density": volume_spec,
            "gravity": gravity_spec,
            "gravity_clean": gravity_spec,
            "observation_height_m": scalar_spec,
            "dx_m": scalar_spec,
            "dy_m": scalar_spec,
            "dz_m": scalar_spec,
        }

    def samples() -> Iterator:
        for path in paths:
            yield load_sample(path, targets=targets, gravity_root=gravity_root,
                              max_abs_density=max_abs_density)

    dataset = tf.data.Dataset.from_generator(
        samples, output_signature=(volume_spec, target_spec)
    ).apply(tf.data.experimental.assert_cardinality(len(paths)))
    if split == "train":
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
