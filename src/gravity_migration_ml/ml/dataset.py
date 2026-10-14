"""Lazy TensorFlow dataset loading for precomputed migration-density pairs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

VOLUME_SHAPE = (24, 64, 64)


def sample_paths(data_root: str | Path, split: str,
                 limit: int | None = None) -> list[Path]:
    """Return deterministic sample paths for one split."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    files = sorted((Path(data_root) / split).glob("*.npz"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No preprocessed {split} samples in {data_root}")
    return files


def load_sample(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized migration input and density target with channel axes."""
    with np.load(path) as archive:
        migration_input = np.asarray(archive["migration_input"], dtype=np.float32)
        density = np.asarray(archive["density"], dtype=np.float32)
    if migration_input.shape != VOLUME_SHAPE or density.shape != VOLUME_SHAPE:
        raise ValueError(
            f"{path}: expected both arrays to have shape {VOLUME_SHAPE}; "
            f"got {migration_input.shape} and {density.shape}"
        )
    if not np.isfinite(migration_input).all() or not np.isfinite(density).all():
        raise ValueError(f"{path}: migration or density contains NaN or infinity")
    if density.min() < 0 or density.max() > 1.0:
        raise ValueError(f"{path}: density targets must be in [0, 1] g/cm^3")
    return migration_input[..., None], density[..., None]


def make_dataset(data_root: str | Path, split: str, batch_size: int = 2,
                 *, limit: int | None = None, seed: int = 42) -> tf.data.Dataset:
    """Create a lazy batched dataset; only the training split is shuffled."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    files = sample_paths(data_root, split, limit)
    path_strings = [str(path) for path in files]

    def samples() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for path in path_strings:
            yield load_sample(path)

    dataset = tf.data.Dataset.from_generator(
        samples,
        output_signature=(
            tf.TensorSpec(shape=(*VOLUME_SHAPE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(*VOLUME_SHAPE, 1), dtype=tf.float32),
        ),
    ).apply(tf.data.experimental.assert_cardinality(len(path_strings)))
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(path_strings), seed=seed, reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
