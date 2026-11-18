"""E11 synthetic-model geometry, metadata, and acquisition checks."""
import numpy as np
import pytest
from src.data.generate_dataset import generate_density, allocate
from src.physics.gravity_forward import (E11_DX_M,E11_DY_M,E11_DZ_M,
                                         observation_coordinates)


def test_domain_dimensions():
    assert E11_DX_M == 125.0
    assert E11_DY_M == 125.0
    assert np.isclose(E11_DZ_M, 1000.0 / 24)
    assert np.isclose(64 * E11_DX_M, 8000)
    assert np.isclose(24 * E11_DZ_M, 1000)
    x,y=observation_coordinates(E11_DX_M,E11_DY_M)
    assert x.shape == y.shape == (81,)
    assert np.allclose(np.diff(x), 125)
    assert np.allclose(np.diff(y), 125)
    assert np.isclose(x[0], -1062.5) and np.isclose(x[-1], 8937.5)


@pytest.mark.parametrize('seed',range(20))
def test_intrusion_geometry(seed):
    rng=np.random.default_rng(seed)
    rho=generate_density('broad_intrusion',rng,density_sign='signed',
                         dx_m=E11_DX_M,dz_m=E11_DZ_M)
    assert rho.shape==(24,64,64)
    assert rho.dtype==np.float32
    assert np.isfinite(rho).all()
    assert rho.max()>0 and rho.min()==0
    assert np.count_nonzero(rho)>50
    # No occupied model boundary voxels.
    assert not rho[0].any() and not rho[-1].any()
    assert not rho[:,:,0].any() and not rho[:,:,-1].any()
    assert not rho[:,0,:].any() and not rho[:,-1,:].any()
    # Should consist of one connected 3-D body (26-connectivity).
    from scipy.ndimage import label
    count=label(rho>0,structure=np.ones((3,3,3),dtype=int))[1]
    assert count==1


def test_original_geometries_preserved():
    for geometry in ('compact','multiple','dipping','salt_dome'):
        rho=generate_density(geometry,np.random.default_rng(17),
                             density_sign='signed')
        assert rho.shape==(24,64,64)
        assert np.any(rho)
    assert np.all(generate_density('salt_dome',np.random.default_rng(17))<=0)


def test_balanced_dataset_allocation():
    for geoms in (('compact','multiple','dipping','salt_dome'),
                  ('compact','multiple','dipping','salt_dome','broad_intrusion')):
        rows=allocate(2400,geoms,[0,5,10,15],np.random.default_rng(42))
        from collections import Counter
        counts=Counter((g,n) for g,n,_ in rows)
        assert len(counts)==4*len(geoms)
        assert len(set(counts.values()))==1


def test_saved_e11_sample_metadata_if_present():
    from pathlib import Path
    files=list(Path('outputs/ml/training/D11_TEST/data').rglob('*.npz'))
    if not files:
        pytest.skip('D11_TEST not generated; create small dataset first')
    with np.load(files[0],allow_pickle=False) as sample:
        assert sample['gravity'].shape==(81,81)
        assert sample['migration_input'].shape==(24,64,64)
        assert np.isclose(sample['dx_m'],125)
        assert np.isclose(sample['dz_m'],1000/24)
        assert float(sample['observation_height_m']) in {0,50,100,150,200}
