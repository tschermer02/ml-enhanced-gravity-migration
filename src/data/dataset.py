"""Load migration-to-density samples for TensorFlow training.

Works with the unified generator (density, migration_input and gravity in one NPZ).
Use targets='density' for standard supervised training, or targets='physics'
for density and observed gravity targets. Also supports older datasets where
observed gravity lives in a separate forward-model folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import numpy as np

# TensorFlow is only required when constructing a tf.data.Dataset.
VOLUME_SHAPE = (24, 64, 64)
GRAVITY_SHAPE = (81, 81)
SPLITS = ("train", "validation", "test")
TargetMode = Literal["density", "physics"]


def sample_paths(data_root: str | Path, split: str,
                 limit: int | None = None) -> list[Path]:
    """Return sorted sample files for a single split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    root = Path(data_root)
    paths = sorted((root / split).glob("*.npz"))
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


def load_sample(path: str | Path, *, targets: TargetMode = "density",
                gravity_root: str | Path | None = None,
                max_abs_density: float | None = 1.0):
    """Return (migration_input, targets) with trailing channel dimensions.

    targets='density': output is the density array.
    targets='physics': output is {'density': density, 'gravity': observed gravity}.

    gravity_root is only needed for older datasets that store gravity separately
    at gravity_root/<split>/<filename>. The unified dataset requires no extra path.
    max_abs_density=None disables the density-range check.
    """
    path = Path(path)
    if targets not in ("density", "physics"):
        raise ValueError("targets must be 'density' or 'physics'")
    with np.load(path, allow_pickle=False) as archive:
        x = _checked_array(archive, "migration_input", VOLUME_SHAPE, path)
        density = _checked_array(archive, "density", VOLUME_SHAPE, path)
        gravity = (_checked_array(archive, "gravity", GRAVITY_SHAPE, path)
                   if targets == "physics" and "gravity" in archive else None)

    if max_abs_density is not None:
        if max_abs_density <= 0:
            raise ValueError("max_abs_density must be positive or None")
        if np.max(np.abs(density)) > max_abs_density + 1e-6:
            raise ValueError(f"{path}: |density| exceeds {max_abs_density} g/cm^3")

    x = x[..., None]
    density = density[..., None]
    if targets == "density":
        return x, density

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
    """Build a lazy, shuffled (train only), batched tf.data.Dataset."""
    import tensorflow as tf

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if targets not in ("density", "physics"):
        raise ValueError("targets must be 'density' or 'physics'")
    paths = sample_paths(data_root, split, limit)
    volume_spec = tf.TensorSpec((*VOLUME_SHAPE, 1), tf.float32)
    target_spec = (volume_spec if targets == "density" else {
        "density": volume_spec,
        "gravity": tf.TensorSpec((*GRAVITY_SHAPE, 1), tf.float32),
    })

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
