import numpy as np
import pytest
from src.prediction.predict import load_prediction_input, run_predictions, sample_paths

class FakeModel:
    def __call__(self, inputs, training=False):
        return np.zeros_like(inputs)

def test_predict_combined_dataset(tmp_path):
    folder = tmp_path / 'data' / 'test'
    folder.mkdir(parents=True)
    density = np.zeros((24, 64, 64), np.float32)
    density[2:5, 3:6, 4:7] = -0.5
    np.savez_compressed(folder / 'sample_000000.npz',
                        density=density,
                        migration_input=np.ones_like(density),
                        gravity=np.zeros((81, 81), np.float32),
                        geometry='dipping', noise_percent=5)
    outputs = tmp_path / 'results'
    rows = run_predictions(FakeModel(), tmp_path / 'data', outputs)
    assert len(rows) == 1
    with np.load(outputs / 'predictions/test/sample_000000.npz') as saved:
        assert saved['prediction'].shape == density.shape
        assert np.min(saved['density']) == -0.5
        assert saved['geometry'].item() == 'dipping'
    with pytest.raises(FileExistsError):
        run_predictions(FakeModel(), tmp_path / 'data', outputs)

def test_missing_migration(tmp_path):
    file = tmp_path / 'bad.npz'
    np.savez(file, density=np.zeros((24,64,64)))
    with pytest.raises(KeyError):
        load_prediction_input(file)
