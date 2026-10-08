"""Rerun Python gravity on all exported validation/test densities; compare against MATLAB."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / 'src'), str(ROOT / 'src' / 'gravity_migration_ml')):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from physics.gravity_forward import GravityForwardModel
except ImportError:
    import physics.gravity_forward as gravity_module
    if hasattr(gravity_module, 'GravityForwardModel'):
        GravityForwardModel = gravity_module.GravityForwardModel
    elif hasattr(gravity_module, 'calculate_gravity'):
        class _CompatGravityForwardModel:
            def __init__(self):
                self.module = gravity_module
            def calculate(self, density):
                return np.asarray(self.module.calculate_gravity(density), dtype=np.float64)
        GravityForwardModel = _CompatGravityForwardModel
    else:
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('analysis_outputs/gravity_forward_validation/inputs'))
    p.add_argument('--matlab-output', type=Path, default=Path('analysis_outputs/gravity_forward_validation/matlab_outputs'))
    p.add_argument('--output', type=Path, default=Path('analysis_outputs/gravity_forward_validation/comparison'))
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--atol-mgal', type=float, default=1e-9)
    p.add_argument('--rtol', type=float, default=1e-9)
    args = p.parse_args()
    operator = GravityForwardModel()
    with (args.inputs / 'index.csv').open(newline='', encoding='utf-8') as f:
        items = list(csv.DictReader(f))
    if args.limit is not None:
        if args.limit < 1: p.error('--limit must be >= 1')
        items = items[:args.limit]
    if not items: raise ValueError('No test examples in index.csv')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []; visual = []
    for i, item in enumerate(items, 1):
        sid = item['sample_id']
        with h5py.File(args.inputs / item['h5_name'], 'r') as f:
            rho = np.asarray(f['density_xfast_gcm3']).reshape(24,64,64)
            stored = np.asarray(f['stored_gz_xfast_mgal']).reshape(81,81)
        python_map = operator.calculate(rho)
        matlab_vec = np.asarray(loadmat(args.matlab_output / f'{sid}.mat')['matlab_gz_xfast_mgal']).reshape(-1)
        if matlab_vec.size != python_map.size: raise ValueError(f'MATLAB receiver count mismatch: {sid}')
        matlab_map = matlab_vec.reshape(81,81)
        delta = python_map - matlab_map
        denom = np.linalg.norm(matlab_map)
        row = {'sample_id':sid,'mae_mgal':float(np.mean(np.abs(delta))),
            'rmse_mgal':float(np.sqrt(np.mean(delta**2))),
            'max_abs_error_mgal':float(np.max(np.abs(delta))),
            'relative_l2':float(np.linalg.norm(delta)/denom) if denom else 0.,
            'correlation':float(np.corrcoef(python_map.ravel(),matlab_map.ravel())[0,1]) if np.std(python_map)>0 and np.std(matlab_map)>0 else None,
            'python_vs_stored_rmse_mgal':float(np.sqrt(np.mean((python_map-stored)**2))),
            'passed':bool(np.allclose(python_map,matlab_map,atol=args.atol_mgal,rtol=args.rtol))}
        rows.append(row)
        visual.append((row['rmse_mgal'], sid, python_map, matlab_map, delta))
        print(f'Compared {i}/{len(items)}: {sid}; max diff {row["max_abs_error_mgal"]:.3e} mGal; pass={row["passed"]}',flush=True)
    with (args.output/'per_sample_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'sample_count':len(rows),'passed_count':sum(r['passed'] for r in rows),'all_passed':all(r['passed'] for r in rows),
        'atol_mgal':args.atol_mgal,'rtol':args.rtol,
        'mean_rmse_mgal':float(np.mean([r['rmse_mgal'] for r in rows])),
        'maximum_absolute_error_mgal':float(max(r['max_abs_error_mgal'] for r in rows))}
    (args.output/'aggregate_summary.json').write_text(json.dumps(summary,indent=2))
    for label, entry in [('best',min(visual)),('worst',max(visual))]:
        rmse,sid,py,mat,err=entry
        fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
        vmin=min(py.min(),mat.min()); vmax=max(py.max(),mat.max())
        for ax,field,title,lo,hi in [(axes[0],py,'Python Gz',vmin,vmax),(axes[1],mat,'MATLAB Gz',vmin,vmax),(axes[2],err,'Python - MATLAB',-max(np.max(np.abs(err)),1e-15),max(np.max(np.abs(err)),1e-15))]:
            im=ax.imshow(field,origin='lower',extent=(-85,715,-85,715),cmap='viridis' if title!='Python - MATLAB' else 'RdBu_r',vmin=lo,vmax=hi)
            ax.set(xlabel='X (m)',ylabel='Y (m)',title=title);fig.colorbar(im,ax=ax,label='mGal')
        fig.suptitle(f'{label}: {sid} | RMSE={rmse:.3e} mGal');fig.savefig(args.output/f'{label}_{sid}.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
    if not summary['all_passed']: sys.exit(1)

if __name__ == '__main__': main()
