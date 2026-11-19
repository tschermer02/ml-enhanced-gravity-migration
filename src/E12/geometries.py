"""Generate E12 four balanced synthetic datasets, using isolated E12 physics.

Run: python -m src.E12.generate --dataset D12_10 --train 2400 --validation 240 --test 480
Small smoke: python -m src.E12.generate --dataset D12_10 --train 10 --validation 5 --test 5
Datasets are separate from E11; never overwrites old work.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .config import (DATASETS, SHAPE, NOISES, MODEL_DX_M, MODEL_DY_M,
                     MODEL_DZ_M, OBS_DX_M, OBS_DY_M)
from .geometries import generate_density
from .physics import forward, migration, normalize_migration
ROOT=Path(__file__).resolve().parents[2]


def assignments(n, geometries, noise_levels, seed):
    if n % len(geometries):
        raise ValueError('Count must be divisible by number of geometries')
    result=[]
    for gi,g in enumerate(geometries):
        for j in range(n//len(geometries)):
            result.append((g,noise_levels[(j+gi)%len(noise_levels)],j))
    np.random.default_rng(seed).shuffle(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--train',type=int,default=2400)
    p.add_argument('--validation',type=int,default=240)
    p.add_argument('--test',type=int,default=480)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-bodies',type=int,default=1)
    p.add_argument('--max-bodies',type=int,default=3)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    geom,heights=DATASETS[args.dataset]
    root=ROOT/'outputs/ml/training'/args.dataset/'data'
    config=dict(dataset=args.dataset,grid='e12',shape=list(SHAPE),
                observation_shape=[81,81],geometries=list(geom),
                noise_percent=list(NOISES),observation_heights_m=list(heights),
                train=args.train,validation=args.validation,test=args.test,seed=args.seed,
                min_bodies=args.min_bodies,max_bodies=args.max_bodies,
                model_dx_m=MODEL_DX_M,model_dy_m=MODEL_DY_M,model_dz_m=MODEL_DZ_M,
                obs_dx_m=OBS_DX_M,obs_dy_m=OBS_DY_M,
                physics='independent staggered-grid 25m FFT point-mass Gzz',
                noise_definition='Gaussian sigma = percent * max(abs(clean Gzz))')
    cfg=root/'config.json'
    if cfg.exists():
        if json.loads(cfg.read_text())!=config:
            raise RuntimeError('Existing dataset configuration differs. Use a different name.')
        if not args.overwrite:
            raise FileExistsError(f'{root} exists; use --overwrite for this matching E12 configuration')
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f'{root} contains data but no config.json')
    root.mkdir(parents=True,exist_ok=True)
    cfg.write_text(json.dumps(config,indent=2))
    for split_idx,(split,n) in enumerate([('train',args.train),('validation',args.validation),('test',args.test)]):
        rows=assignments(n,geom,NOISES,args.seed+split_idx)
        for j,(g,noise,group) in enumerate(rows):
            height=heights[(group+geom.index(g))%len(heights)]
            path=root/split/g/f'noise_{int(noise):02d}'/f'sample_{j:06d}.npz'
            if path.exists() and not args.overwrite:
                raise FileExistsError(path)
            seq=np.random.SeedSequence([args.seed,split_idx,geom.index(g),group])
            rho=generate_density(g,np.random.default_rng(seq),
                 min_bodies=args.min_bodies,max_bodies=args.max_bodies,
                 density_sign='signed',dx_m=MODEL_DX_M,dz_m=MODEL_DZ_M)
            clean=forward(rho,height)
            sigma=float(noise/100.*np.max(np.abs(clean)))
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,split_idx,j,919]))
            obs=(clean+rng.normal(0,sigma,clean.shape)).astype(np.float32)
            migrated=migration(obs,height)
            mig_norm,scale=normalize_migration(migrated)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,density=rho,gravity=obs,gravity_clean=clean,
                    migration=migrated,migration_input=mig_norm,migration_scale=scale,
                    observation_height_m=np.float32(height),geometry=np.str_(g),
                    noise_percent=np.float32(noise),noise_std=np.float32(sigma),
                    model_dx_m=MODEL_DX_M,model_dy_m=MODEL_DY_M,
                    model_dz_m=MODEL_DZ_M,obs_dx_m=OBS_DX_M,obs_dy_m=OBS_DY_M,
                    observation_component=np.str_('GZZ'),observation_unit=np.str_('E'))
            if (j+1)%25==0 or j+1==n:
                print(f'{args.dataset} {split}: {j+1}/{n}',flush=True)
    print('Saved',root)

if __name__=='__main__':main()
