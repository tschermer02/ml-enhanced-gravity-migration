"""E07: four signed-density geometries, with paired 0/5/10/15% gravity noise.

Training and validation have balanced geology/noise conditions. Held-out test
volumes are IDENTICAL across all four noise levels within each geology.
Noise std = percentage of maximum absolute CLEAN gravity anomaly (mGal).

Run from repository root, using a CHPC compute allocation for --stage physics.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'src',ROOT/'scripts'):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
from generate_e05a_f_cases import make_density as original_density
from physics.gravity_forward import calculate_gravity
from physics.gravity_migration import calculate_migration

CASES={'compact':'E05A','multiple':'E05B','dipping':'E05C','salt_dome':'E05D'}
LEVELS=(0,5,10,15)
SPLIT_OFFSET={'train':1001,'validation':2002,'test':3003}


def make_density(case,seed,index):
    result=np.asarray(original_density(CASES[case],seed,index),dtype=np.float32)
    if case=='salt_dome':result=-result
    if result.shape!=(24,64,64) or not np.isfinite(result).all() or not np.any(result):
        raise ValueError(f'Bad model {case} {index}')
    if np.max(np.abs(result))>1 or (case=='salt_dome' and np.max(result)>0):
        raise ValueError(f'Bad density range {case} {index}')
    return result


def assignments(split,count,seed):
    """Train/val: independent randomized conditions; test: paired noise on identical bodies."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,SPLIT_OFFSET[split],71]))
    rows=[]
    if split=='test':
        for case in CASES:
            for index in range(count):
                for noise in LEVELS:
                    rows.append(dict(sample_id=f'{case}_{index:06d}_n{noise:02d}',
                                     case=case,case_index=index,noise_percent=noise,split=split))
    else:
        conditions=[(case,noise) for case in CASES for noise in LEVELS]
        if count%len(conditions):
            raise ValueError(f'{split} count must be divisible by {len(conditions)} to balance all 16 conditions')
        indices={case:0 for case in CASES}
        for case,noise in conditions*(count//len(conditions)):
            idx=indices[case];indices[case]+=1
            rows.append(dict(sample_id=f'sample_{len(rows):06d}',case=case,
                             case_index=idx,noise_percent=noise,split=split))
        rng.shuffle(rows)
        for idx,row in enumerate(rows):row['sample_id']=f'sample_{idx:06d}'
    return rows


def create_models(dataset_root,rows,seed,overwrite):
    for k,row in enumerate(rows,1):
        path=dataset_root/'synthetic_models'/row['split']/(row['sample_id']+'.npz')
        if path.exists() and not overwrite:
            with np.load(path) as f:
                if str(f['case'].item())!=row['case'] or int(f['case_index'])!=row['case_index'] or int(f['noise_percent'])!=row['noise_percent']:
                    raise ValueError(f'Existing sample metadata differs: {path}; specify --overwrite')
            continue
        # Separate split seeds, so geology cannot leak between train/val/test.
        model_seed=seed+SPLIT_OFFSET[row['split']]
        density=make_density(row['case'],model_seed,row['case_index'])
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,density=density,case=row['case'],case_index=row['case_index'],
                            noise_percent=row['noise_percent'],split=row['split'],cell_size_m=10.0)
        if k%100==0 or k==len(rows):print(f'Models {k}/{len(rows)}',flush=True)


def physics_job(job):
    source,fwd_path,data_path,seed,overwrite=job
    source,fwd_path,data_path=map(Path,(source,fwd_path,data_path))
    if fwd_path.exists() and data_path.exists() and not overwrite:return 'skip'
    with np.load(source) as f:
        density=np.asarray(f['density'],dtype=np.float32)
        case=str(f['case'].item());index=int(f['case_index'])
        noise=int(f['noise_percent']);split=str(f['split'].item())
    # For test data each noise variant of an underlying geology shares a clean
    # forward model and a standardized noise realization, scaled by percentage.
    clean=np.asarray(calculate_gravity(density),dtype=np.float64)
    std=noise/100*float(np.max(np.abs(clean)))
    measured=clean.copy()
    if noise:
        case_idx=list(CASES).index(case)
        rng=np.random.default_rng(np.random.SeedSequence([seed,SPLIT_OFFSET[split],case_idx,index,991]))
        measured+=std*rng.standard_normal(clean.shape)
    migration=np.asarray(calculate_migration(measured,restrict=False),dtype=np.float64)
    scale=max(float(np.max(np.abs(migration))),1e-8)
    normalized=(migration/scale).astype(np.float32)
    if not np.isfinite(normalized).all():raise ValueError(f'Nonfinite migration {source}')
    fwd_path.parent.mkdir(parents=True,exist_ok=True)
    data_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(fwd_path,gravity=measured.astype(np.float32),gravity_clean=clean.astype(np.float32),noise_std_mgal=std)
    np.savez_compressed(data_path,density=density,migration=migration.astype(np.float32),migration_input=normalized,migration_scale=scale)
    return 'ok'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['models','physics','all'],default='all')
    p.add_argument('--train',type=int,default=2000)
    p.add_argument('--validation',type=int,default=160)
    p.add_argument('--test-per-case',type=int,default=25,help='Geology models per category, each reused across all 4 noise levels')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dataset-root',type=Path,default=Path('src/data/dataset_e07'))
    p.add_argument('--data-root',type=Path,default=Path('outputs/ml/training/E07/data'))
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if min(args.train,args.validation,args.test_per_case,args.workers)<1:p.error('Counts and workers must be positive')
    base=args.dataset_root if args.dataset_root.is_absolute() else ROOT/args.dataset_root
    data=args.data_root if args.data_root.is_absolute() else ROOT/args.data_root
    rows=[]
    for split,n in [('train',args.train),('validation',args.validation),('test',args.test_per_case)]:
        rows.extend(assignments(split,n,args.seed))
    manifest=base/'manifest.csv'
    if args.stage in ('models','all'):
        create_models(base,rows,args.seed,args.overwrite)
        manifest.parent.mkdir(parents=True,exist_ok=True)
        with manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        if not manifest.is_file():raise FileNotFoundError(f'Missing manifest: {manifest}; run --stage models')
        with manifest.open(newline='') as f: saved=list(csv.DictReader(f))
        if [(r['sample_id'],r['case'],int(r['noise_percent'])) for r in saved] != [(r['sample_id'],r['case'],r['noise_percent']) for r in rows]:
            raise ValueError('Manifest differs from requested seed/counts; use original settings')
    if args.stage in ('physics','all'):
        jobs=[]
        for row in rows:
            name=row['sample_id']+'.npz';split=row['split']
            source=base/'synthetic_models'/split/name
            if not source.is_file():raise FileNotFoundError(source)
            jobs.append((str(source),str(base/'forward_models'/split/name),str(data/split/name),args.seed,args.overwrite))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures=[ex.submit(physics_job,job) for job in jobs]
            for index,future in enumerate(as_completed(futures),1):
                future.result()
                if index%25==0 or index==len(jobs):print(f'Physics {index}/{len(jobs)}',flush=True)
    base.mkdir(parents=True,exist_ok=True)
    (base/'dataset_config.json').write_text(json.dumps(dict(experiment='E07',seed=args.seed,train=args.train,
        validation=args.validation,test_per_geology=args.test_per_case,test_total=args.test_per_case*16,
        geologies=list(CASES),noise_percent=list(LEVELS),noise_definition='Gaussian std (% max abs clean gravity)',
        target='signed density g/cm3, salt dome negative',train_noise='balanced 0/5/10/15',
        test_noise='paired identical density, same standardized noise vector per model'),indent=2))
    print('E07 generation complete',flush=True)

if __name__=='__main__':main()
