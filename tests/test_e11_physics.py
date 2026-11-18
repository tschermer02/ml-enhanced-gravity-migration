"""E11 anisotropic Gzz numerical, height, migration and gradient tests."""
import numpy as np
import pytest
from src.physics.gravity_forward import (calculate_gzz,calculate_gravity,
    E11_DX_M,E11_DY_M,E11_DZ_M)
from src.physics.gravity_migration import calculate_migration,normalize_migration

GRID=dict(dx_m=E11_DX_M,dy_m=E11_DY_M,dz_m=E11_DZ_M)


def body():
    rho=np.zeros((24,64,64),np.float32)
    rho[3:8,28:36,25:38]=.45
    return rho


def test_zero_density():
    assert not calculate_gzz(np.zeros((24,64,64)),**GRID).any()


def test_legacy_defaults():
    rho=body()
    np.testing.assert_allclose(calculate_gzz(rho),calculate_gzz(
        rho,dx_m=10,dy_m=10,dz_m=10,observation_height_m=0),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('height',[0,50,100,150,200])
def test_forward_migration(height):
    rho=body()
    gzz=calculate_gravity(rho,component='gzz',observation_height_m=height,**GRID)
    migration=calculate_migration(gzz,component='gzz',restrict=False,
                                  observation_height_m=height,**GRID)
    normalized,scale=normalize_migration(migration)
    assert gzz.shape==(81,81)
    assert migration.shape==(24,64,64)
    assert np.isfinite(gzz).all() and np.isfinite(migration).all()
    assert scale>0
    assert np.max(np.abs(normalized))<=1.000001


def test_height_weakens_central_anomaly():
    rho=body()
    fields=[calculate_gzz(rho,observation_height_m=h,**GRID) for h in (0,100,200)]
    # Central maximum of compact positive body should decay with elevation.
    assert fields[0].max()>fields[1].max()>fields[2].max()


def test_tf_gzz_matches_numpy_and_has_gradient():
    tf=pytest.importorskip('tensorflow')
    from src.physics.gravity_forward_tf import gravity_gzz_forward_tf
    rho=body()
    batch=np.stack((rho,-rho)).astype(np.float32)
    heights=np.array([0.,100.],np.float32)
    tf_result=gravity_gzz_forward_tf(tf.constant(batch[...,None]),
                      observation_height_m=heights,**GRID).numpy()[...,0]
    for i,h in enumerate(heights):
        ref=calculate_gzz(batch[i],observation_height_m=float(h),**GRID)
        np.testing.assert_allclose(tf_result[i],ref,rtol=1e-4,atol=1e-3)
    var=tf.Variable(batch[...,None])
    with tf.GradientTape() as tape:
        gzz=gravity_gzz_forward_tf(var,observation_height_m=heights,**GRID)
        loss=tf.reduce_mean(gzz*gzz)
    gradients=tape.gradient(loss,var)
    assert gradients is not None
    assert np.isfinite(gradients.numpy()).all()
    assert np.any(np.abs(gradients.numpy())>0)


def test_e11_loss_physics_gradient():
    tf=pytest.importorskip('tensorflow')
    from src.losses.losses import e11_loss_components
    from src.physics.gravity_forward_tf import gravity_gzz_forward_tf
    rho=body()[None,...,None]
    truth=tf.constant(rho)
    observed=gravity_gzz_forward_tf(truth,observation_height_m=100.,**GRID)
    variable=tf.Variable(rho*0.8)
    with tf.GradientTape() as tape:
        parts=e11_loss_components(truth,variable,gravity_clean=observed,
                                  observation_height_m=100.,physics_weight=.01,
                                  physics_normalization_e=100.,**GRID)
    g=tape.gradient(parts['total'],variable)
    assert g is not None and np.isfinite(g.numpy()).all()
    assert np.isfinite(float(parts['total']))
    assert float(parts['physics'])>0
