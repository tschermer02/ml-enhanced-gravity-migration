import numpy as np
from src.physics.gravity_forward import calculate_gravity, calculate_gz, calculate_gzz
from src.physics.gravity_migration import calculate_migration, calculate_gzz_migration, normalize_migration


def test_zero():
    density = np.zeros((24, 64, 64))
    assert np.allclose(calculate_gravity(density), 0)
    assert np.allclose(calculate_gravity(density, 'gzz'), 0)


def test_dispatch_and_normalize():
    density = np.zeros((24, 64, 64))
    density[3:5, 27:33, 27:33] = .5
    for component, direct in [('gz', calculate_gz), ('gzz', calculate_gzz)]:
        data = calculate_gravity(density, component)
        np.testing.assert_array_equal(data, direct(density))
        migrated = calculate_migration(data, component)
        assert migrated.shape == density.shape
        normalized, scale = normalize_migration(migrated)
        assert scale > 0 and np.max(np.abs(normalized)) <= 1.000001
    np.testing.assert_allclose(calculate_gzz_migration(calculate_gzz(density)), calculate_migration(calculate_gzz(density), 'gzz'))
