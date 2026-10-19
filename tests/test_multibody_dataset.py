"""Tests for the multi-body synthetic density generator.

Run from the repository root:
    python -m pytest tests/test_multibody_dataset.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load the generator by its file path; src/data is not a Python package.
GENERATOR_PATH = SRC / "data" / "generate_dataset_multibody.py"
spec = importlib.util.spec_from_file_location("generate_dataset_multibody", GENERATOR_PATH)
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

from physics.gravity_forward import calculate_gravity


def assert_valid_model(density, bounds, contrasts, min_count=1, max_count=3, gap=2):
    """Check shape, placement, per-body density, and minimum separation."""
    assert density.shape == (24, 64, 64)
    assert density.dtype == np.float32
    assert np.all(np.isfinite(density))
    assert 0 <= density.min() <= density.max() <= 1.0

    n = len(bounds)
    assert min_count <= n <= max_count
    assert bounds.shape == (n, 6)
    assert contrasts.shape == (n,)
    assert np.all((contrasts >= generator.DENSITY_MIN) &
                  (contrasts <= generator.DENSITY_MAX))

    expected = np.zeros_like(density)
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(bounds):
        assert 0 <= x0 < x1 <= generator.NX
        assert 0 <= y0 < y1 <= generator.NY
        assert 0 <= z0 < z1 <= generator.NZ
        assert generator.WIDTH_MIN <= x1 - x0 <= generator.WIDTH_MAX
        assert generator.WIDTH_MIN <= y1 - y0 <= generator.WIDTH_MAX
        assert generator.THICKNESS_MIN <= z1 - z0 <= generator.THICKNESS_MAX
        assert generator.TOP_MIN <= z0 <= generator.TOP_MAX
        assert x0 >= generator.LATERAL_MARGIN_CELLS
        assert y0 >= generator.LATERAL_MARGIN_CELLS
        assert x1 <= generator.NX - generator.LATERAL_MARGIN_CELLS
        assert y1 <= generator.NY - generator.LATERAL_MARGIN_CELLS

        for j in range(i):
            assert generator.separated(tuple(bounds[i]), tuple(bounds[j]), gap)

        expected[z0:z1, y0:y1, x0:x1] = contrasts[i]

    np.testing.assert_array_equal(density, expected)
    assert np.count_nonzero(density) > 0


@pytest.mark.parametrize("count", [1, 2, 3])
def test_requested_body_counts_and_geometry(count):
    rng = np.random.default_rng(100 + count)
    for _ in range(5):
        density, bounds, contrasts = generator.generate_density(
            rng, min_bodies=count, max_bodies=count, gap_cells=2
        )
        assert_valid_model(density, bounds, contrasts, count, count, gap=2)


def test_random_body_count_within_range():
    rng = np.random.default_rng(42)
    observed = set()
    for _ in range(30):
        density, bounds, contrasts = generator.generate_density(rng)
        observed.add(len(bounds))
        assert_valid_model(density, bounds, contrasts)
    assert observed == {1, 2, 3}


def test_reproducible_generation():
    a = generator.generate_density(np.random.default_rng(123))
    b = generator.generate_density(np.random.default_rng(123))
    for a_value, b_value in zip(a, b):
        np.testing.assert_array_equal(a_value, b_value)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        generator.generate_density(np.random.default_rng(42), min_bodies=0)
    with pytest.raises(ValueError):
        generator.generate_density(np.random.default_rng(42), min_bodies=3, max_bodies=2)
    with pytest.raises(ValueError):
        generator.generate_density(np.random.default_rng(42), gap_cells=-1)


def test_saved_npz_metadata_and_no_overwrite(tmp_path):
    root = tmp_path / "synthetic_models"
    options = dict(
        split="train", sample_count=3, output_root=root, seed=12345,
        min_bodies=2, max_bodies=3, gap_cells=2,
    )
    generator.generate_split(**options)

    paths = sorted((root / "train").glob("sample_*.npz"))
    assert len(paths) == 3

    for path in paths:
        with np.load(path) as archive:
            assert {
                "density", "n_bodies", "prism_bounds_cells",
                "prism_bounds_m", "prism_densities_gcm3",
                "cell_size_m", "seed",
            } <= set(archive.files)

            density = archive["density"]
            bounds = archive["prism_bounds_cells"]
            contrasts = archive["prism_densities_gcm3"]
            assert int(archive["n_bodies"]) == len(bounds)
            assert int(archive["seed"]) == 12345
            assert float(archive["cell_size_m"]) == generator.CELL_SIZE_M
            np.testing.assert_array_equal(
                archive["prism_bounds_m"], bounds * generator.CELL_SIZE_M
            )
            assert_valid_model(density, bounds, contrasts, 2, 3, gap=2)

    with pytest.raises(FileExistsError):
        generator.generate_split(**options)


def test_split_reproducibility(tmp_path):
    common = dict(
        split="validation", sample_count=3, seed=978,
        min_bodies=1, max_bodies=3, gap_cells=2,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    generator.generate_split(output_root=first, **common)
    generator.generate_split(output_root=second, **common)

    for path in sorted((first / "validation").glob("*.npz")):
        counterpart = second / "validation" / path.name
        with np.load(path) as a, np.load(counterpart) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_gravity_forward_superposition():
    """Forward modeling adds contributions from separate bodies."""
    first = np.zeros((24, 64, 64), dtype=np.float32)
    second = np.zeros_like(first)
    first[3, 14, 12] = 0.4
    second[9, 38, 40] = 0.7

    gravity_combined = calculate_gravity(first + second)
    gravity_individual = calculate_gravity(first) + calculate_gravity(second)

    assert gravity_combined.shape == (81, 81)
    assert np.isfinite(gravity_combined).all()
    np.testing.assert_allclose(
        gravity_combined, gravity_individual, rtol=1e-10, atol=1e-12
    )
