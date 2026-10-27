"""Numerical checks for MATLAB-equivalent Gzz physics. Run from repo root."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from physics.gravity_gradient_forward import calculate_gzz, G, CELL_SIZE
from physics.gravity_gradient_migration import calculate_gzz_migration, GAMMA


def main():
    model=np.zeros((24,64,64),dtype=np.float64)
    model[4,21,29]=-.4
    model[12,9,50]=.6
    observed=calculate_gzz(model)
    yy,xx=np.meshgrid(np.linspace(-85,715,81),np.linspace(-85,715,81),indexing='ij')
    direct=np.zeros_like(observed)
    for iz,iy,ix in zip(*np.nonzero(model)):
        dx=(ix+.5)*CELL_SIZE-xx
        dy=(iy+.5)*CELL_SIZE-yy
        dz=(iz+.5)*CELL_SIZE
        r2=dx*dx+dy*dy+dz*dz
        direct+=G*model[iz,iy,ix]*CELL_SIZE**3*1e12*(3*dz*dz-r2)/(r2**2*np.sqrt(r2))
    np.testing.assert_allclose(observed,direct,rtol=1e-10,atol=1e-10)
    print('PASS: FFT Gzz matches direct MATLAB case-7 kernel.')
    rng=np.random.default_rng(7)
    test_obs=rng.normal(size=(81,81))
    migrated=calculate_gzz_migration(test_obs)
    for iz,iy,ix in ((0,0,0),(5,25,36),(23,63,63)):
        dx=xx-(ix+.5)*CELL_SIZE
        dy=yy-(iy+.5)*CELL_SIZE
        dz=(iz+.5)*CELL_SIZE
        r2=dx*dx+dy*dy+dz*dz
        direct_mig=np.sum(test_obs*(3*dz*dz-r2)/(r2**2*np.sqrt(r2)))
        direct_mig*=dz**2/(GAMMA*np.sqrt(3*np.pi)/2)
        np.testing.assert_allclose(migrated[iz,iy,ix],direct_mig,rtol=1e-10,atol=1e-9)
    print('PASS: FFT migration matches MATLAB case-7 weighted sum.')
    # Verify the UNWEIGHTED mathematical adjoint, before MATLAB depth weighting.
    # A = G*volume*1e12 * kernel; M = (depth^2 / czz) * (K.T @ data).
    sample=rng.normal(size=(24,64,64))
    data=rng.normal(size=(81,81))
    lhs=np.vdot(calculate_gzz(sample),data)
    back=calculate_gzz_migration(data)
    depths=(np.arange(24)+.5)*CELL_SIZE
    czz=GAMMA*np.sqrt(3*np.pi)/2
    adjoint=back*(czz/depths[:,None,None]**2)*(G*CELL_SIZE**3*1e12)
    rhs=np.vdot(sample,adjoint)
    np.testing.assert_allclose(lhs,rhs,rtol=1e-10,atol=1e-7)
    print('PASS: unweighted discrete adjoint dot-product test.')

if __name__=='__main__': main()
