"""Compare Python/MATLAB Gz,Gzz forward and migration outputs on 100 samples."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from scipy.io import loadmat

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/ml/validation/matlab_D01'))
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--rtol',type=float,default=1e-7)
    p.add_argument('--atol',type=float,default=1e-9)
    a=p.parse_args();py=sorted((a.root/'python').glob('sample_*.npz'))[:a.limit]
    if len(py)!=a.limit:raise ValueError(f'Expected {a.limit} Python results, got {len(py)}')
    rows=[]
    for i,path in enumerate(py,1):
        mpath=a.root/'matlab'/(path.stem+'.mat')
        if not mpath.is_file():raise FileNotFoundError(mpath)
        m=loadmat(mpath)
        with np.load(path) as z:
            for comp in ('gz','gzz'):
                for method in ('forward','migration'):
                    key=f'{comp}_{method}';x=z[key];y=m[key]
                    if x.shape!=y.shape:raise ValueError(f'{path.stem} {key} {x.shape} != {y.shape}')
                    diff=x-y
                    rows.append(dict(sample=path.stem,component=comp,operator=method,
                        rmse=float(np.sqrt(np.mean(diff**2))),max_abs=float(np.max(np.abs(diff))),
                        relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(y),1e-30)),
                        passed=bool(np.allclose(x,y,rtol=a.rtol,atol=a.atol))))
        print(f'Compared {i}/{len(py)}: {path.stem}',flush=True)
    with (a.root/'comparison_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={}
    for comp in ('gz','gzz'):
        for method in ('forward','migration'):
            rr=[r for r in rows if r['component']==comp and r['operator']==method]
            summary[f'{comp}_{method}']={'samples':len(rr),'passed':sum(r['passed'] for r in rr),
                'mean_rmse':float(np.mean([r['rmse'] for r in rr])),
                'worst_max_abs':float(max(r['max_abs'] for r in rr)),
                'worst_relative_l2':float(max(r['relative_l2'] for r in rr))}
    (a.root/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not all(r['passed'] for r in rows):raise SystemExit('Some comparisons failed. See comparison_metrics.csv')
if __name__=='__main__':main()
