"""Export 100 D01 test densities and compute Python reference maps/volumes."""
from pathlib import Path
import argparse, csv, numpy as np
from scipy.io import savemat
from src.physics.gravity_forward import calculate_gravity
from src.physics.gravity_migration import calculate_migration

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=Path('outputs/ml/training/D01/data'))
    p.add_argument('--output',type=Path,default=Path('outputs/ml/validation/matlab_D01'))
    p.add_argument('--limit',type=int,default=100)
    a=p.parse_args(); files=sorted((a.data_root/'test').glob('*.npz'))[:a.limit]
    if len(files)!=a.limit: raise ValueError(f'Expected {a.limit} D01 test files; found {len(files)}')
    ins=a.output/'inputs'; refs=a.output/'python'; ins.mkdir(parents=True,exist_ok=True);refs.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,f in enumerate(files,1):
        with np.load(f) as z: rho=np.asarray(z['density'],dtype=np.float64)
        if rho.shape!=(24,64,64): raise ValueError(f'Invalid density shape {f}: {rho.shape}')
        savemat(ins/(f.stem+'.mat'),{'density':rho},do_compression=True)
        arrays={}
        for comp in ('gz','gzz'):
            gravity=calculate_gravity(rho,component=comp)
            migration=calculate_migration(gravity,component=comp,restrict=False)
            arrays[f'{comp}_forward']=gravity
            arrays[f'{comp}_migration']=migration
        np.savez_compressed(refs/(f.stem+'.npz'),**arrays)
        rows.append((f.stem,str(f)))
        print(f'Python {i}/{len(files)}: {f.stem}',flush=True)
    with (a.output/'samples.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(('sample_id','source'));w.writerows(rows)
    print('Output:',a.output)
if __name__=='__main__':main()
