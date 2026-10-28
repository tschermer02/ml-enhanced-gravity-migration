"""Lightweight checks for the unified dataset loader (no TensorFlow required)."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_sample, sample_paths


class DatasetTests(unittest.TestCase):
    def test_combined_signed_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "train"
            folder.mkdir()
            density = np.zeros((24, 64, 64), np.float32)
            density[2:4, 5:8, 7:10] = -0.5
            path = folder / "sample_000000.npz"
            np.savez_compressed(path, density=density,
                                migration_input=np.zeros_like(density),
                                gravity=np.ones((81, 81), np.float32))
            self.assertEqual(sample_paths(root, "train"), [path])
            x, y = load_sample(path)
            self.assertEqual(x.shape, (24, 64, 64, 1))
            self.assertEqual(y.shape, (24, 64, 64, 1))
            self.assertAlmostEqual(float(y.min()), -0.5)
            _, targets = load_sample(path, targets="physics")
            self.assertEqual(targets["gravity"].shape, (81, 81, 1))
            self.assertAlmostEqual(float(targets["gravity"].mean()), 1.0)

    def test_legacy_separate_gravity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "test").mkdir(parents=True)
            (root / "forward" / "test").mkdir(parents=True)
            path = root / "data" / "test" / "sample_000000.npz"
            np.savez_compressed(path, density=np.zeros((24, 64, 64)),
                                migration_input=np.zeros((24, 64, 64)))
            np.savez_compressed(root / "forward" / "test" / path.name,
                                gravity=np.zeros((81, 81)))
            _, y = load_sample(path, targets="physics", gravity_root=root / "forward")
            self.assertEqual(y["gravity"].shape, (81, 81, 1))


if __name__ == "__main__":
    unittest.main()
