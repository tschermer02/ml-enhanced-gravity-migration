"""Generate a SHARED regional dataset for E10A and E10B.

Example:
 python -m src.data.generate_e10 --output-root /scratch/general/vast/$USER/e10/data
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from src.data.e10_regional import (GEOMETRIES,SHAPE,OBS_SHAPE,DX,DZ,FLIGHT_HEIGHT,
                                   make_density,forward_gzz,migration_gzz)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--train',type=int,default=2000)
    p.add_argument('--validation',type=int,default=100)
    p.add_argument('--test',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--noise',nargs='+',type=float,default=[0,5,10,15])
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if any(x<4 or x%4 for x in (args.train,args.validation,args.test)):
        p.error('Each split must be a positive multiple of four (four geometry classes)')
    if not args.noise or min(args.noise)<0: p.error('Specify nonnegative noise levels')
    out=args.output_root.expanduser().resolve()
    config={'train':args.train,'validation':args.validation,'test':args.test,
            'seed':args.seed,'noise':args.noise,'shapes':{'density':SHAPE,'gravity':OBS_SHAPE},
            'cell_m':[DX,DX,DZ],'flight_height_m':FLIGHT_HEIGHT,
            'density_unit':'g/cm3','gravity_unit':'Eotvos',
            'input_convention':'raw migration; normalize only at training time'}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != json.loads(json.dumps(config)):
        raise RuntimeError('Existing config differs; select another dataset directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    records=[]
    for split_i,(split,count) in enumerate([('train',args.train),('validation',args.validation),('test',args.test)]):
        for geom_i,geom in enumerate(GEOMETRIES):
            for k in range(count//4):
                path=out/split/geom/f'sample_{k:05d}.npz'
                noise_pct=args.noise[k%len(args.noise)]
                records.append((split,geom,k,noise_pct,str(path)))
                if path.exists() and not args.overwrite: continue
                rng=np.random.default_rng(np.random.SeedSequence([args.seed,split_i,geom_i,k]))
                rho=make_density(geom,rng)
                clean=forward_gzz(rho)
                sigma=float(np.max(np.abs(clean)))*noise_pct/100
                measured=(clean+sigma*rng.standard_normal(clean.shape)).astype(np.float32)
                mig=migration_gzz(measured)
                if not np.isfinite(mig).all(): raise RuntimeError(f'Non-finite migration in {path}')
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,density=rho,gravity=measured,migration=mig,
                                    migration_scale=np.float64(np.max(np.abs(mig))),
                                    geometry=np.str_(geom),noise_percent=np.float32(noise_pct))
                if (k+1)%25==0 or k+1==count//4: print(split,geom,k+1,'/',count//4,flush=True)
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','geometry','index','noise_percent','path']);w.writerows(records)
    print('Dataset complete at:',out,flush=True)

if __name__=='__main__':main()
