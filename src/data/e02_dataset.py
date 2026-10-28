"""E02 data loading with E01 migration inputs and observed gravity targets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

from gravity_migration_ml.ml.dataset import VOLUME_SHAPE, sample_paths

GRAVITY_SHAPE = (81, 81)


def load_e02_sample(migration_path: str | Path,
                    gravity_root: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the unchanged E01 input/target plus observed gravity from its source stage."""
    migration_path = Path(migration_path)
    gravity_path = Path(gravity_root) / migration_path.parent.name / migration_path.name
    with np.load(migration_path) as archive:
        migration_input = np.asarray(archive["migration_input"], dtype=np.float32)
        density = np.asarray(archive["density"], dtype=np.float32)
    if not gravity_path.is_file():
        raise FileNotFoundError(f"Observed gravity file is missing: {gravity_path}")
    with np.load(gravity_path) as archive:
        gravity = np.asarray(archive["gravity"], dtype=np.float32)

    if migration_input.shape != VOLUME_SHAPE or density.shape != VOLUME_SHAPE:
        raise ValueError(f"{migration_path}: E01 volumes must have shape {VOLUME_SHAPE}")
    if gravity.shape != GRAVITY_SHAPE:
        raise ValueError(f"{gravity_path}: observed gravity must have shape {GRAVITY_SHAPE}")
    if (not np.isfinite(migration_input).all()
            or not np.isfinite(density).all()
            or not np.isfinite(gravity).all()):
        raise ValueError(f"{migration_path}: E02 data contain NaN or infinity")
    return migration_input[..., None], {
        "density": density[..., None],
        "gravity": gravity[..., None],
    }


def make_e02_dataset(data_root: str | Path, gravity_root: str | Path,
                     split: str, batch_size: int = 2, *,
                     limit: int | None = None, seed: int = 42) -> tf.data.Dataset:
    """Create a lazy dataset yielding migration and both E02 supervision targets."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    files = sample_paths(data_root, split, limit)
    migration_paths = [str(path) for path in files]
    gravity_root_text = str(gravity_root)

    def samples() -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        for path in migration_paths:
            yield load_e02_sample(path, gravity_root_text)

    dataset = tf.data.Dataset.from_generator(
        samples,
        output_signature=(
            tf.TensorSpec(shape=(*VOLUME_SHAPE, 1), dtype=tf.float32),
            {
                "density": tf.TensorSpec(shape=(*VOLUME_SHAPE, 1), dtype=tf.float32),
                "gravity": tf.TensorSpec(shape=(*GRAVITY_SHAPE, 1), dtype=tf.float32),
            },
        ),
    ).apply(tf.data.experimental.assert_cardinality(len(migration_paths)))
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(migration_paths), seed=seed, reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
