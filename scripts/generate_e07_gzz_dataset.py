"""E07_Gzz: reproduce E07 density/noise splits using MATLAB-style Gzz physics.

Generate an independent data-root so original gz/E07 remains unchanged.
  python scripts/generate_e07_gzz_dataset.py --stage models
  python scripts/generate_e07_gzz_dataset.py --stage physics --workers 4

Gzz measurements are Eotvos (E). Gaussian noise std is 0/5/10/15% of
max(abs(clean_Gzz)). All four noise variants of each test target share the
same clean signal and one standardized noise realization.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'src', ROOT/'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from generate_e07_dataset import assignments, make_density, CASES, LEVELS, SPLIT_OFFSET
from physics.gravity_gradient_forward import calculate_gzz
from physics.gravity_gradient_migration import calculate_gzz_migration


def physics_one(job):
    source, observed_file, prepared_file, seed, overwrite = job
    source, observed_file, prepared_file = map(Path, (source, observed_file, prepared_file))
    if observed_file.exists() and prepared_file.exists() and not overwrite:
        return 'skipped'
    with np.load(source) as f:
        density = np.asarray(f['density'], dtype=np.float32)
        case = str(f['case'].item())
        index = int(f['case_index'])
        noise = int(f['noise_percent'])
        split = str(f['split'].item())
    clean = calculate_gzz(density)
    sigma = noise * .01 * float(np.max(np.abs(clean)))
    observed = clean.copy()
    if noise:
        case_idx = list(CASES).index(case)
        rng = np.random.default_rng(np.random.SeedSequence([
            seed, SPLIT_OFFSET[split], case_idx, index, 991]))
        observed += sigma * rng.standard_normal(clean.shape)
    migration = calculate_gzz_migration(observed, restrict=False)
    scale = max(float(np.max(np.abs(migration))), 1e-8)
    normalized = (migration/scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f'non-finite migrated volume: {source}')
    observed_file.parent.mkdir(parents=True, exist_ok=True)
    prepared_file.parent.mkdir(parents=True, exist_ok=True)
    # 'gravity' retained for compatibility with other data readers, but now E.
    np.savez_compressed(observed_file, gravity=observed.astype(np.float32),
                        gravity_clean=clean.astype(np.float32), noise_std_eotvos=sigma,
                        observed_component='Gzz', observed_unit='E')
    np.savez_compressed(prepared_file, density=density,
                        migration=migration.astype(np.float32),
                        migration_input=normalized, migration_scale=np.float64(scale),
                        observed_component='Gzz')
    return 'created'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['models','physics','all'],default='all')
    p.add_argument('--train',type=int,default=2000)
    p.add_argument('--validation',type=int,default=160)
    p.add_argument('--test-per-case',type=int,default=25)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dataset-root',type=Path,default=Path('src/data/dataset_e07_gzz'))
    p.add_argument('--data-root',type=Path,default=Path('outputs/ml/training/E07_Gzz/data'))
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if min(args.train,args.validation,args.test_per_case,args.workers)<1:
        p.error('Counts/workers must be positive')
    droot=args.dataset_root if args.dataset_root.is_absolute() else ROOT/args.dataset_root
    proot=args.data_root if args.data_root.is_absolute() else ROOT/args.data_root
    rows=[]
    for split,n in [('train',args.train),('validation',args.validation),('test',args.test_per_case)]:
        rows.extend(assignments(split,n,args.seed))
    manifest=droot/'manifest.csv'
    if args.stage in ('models','all'):
        for i,row in enumerate(rows,1):
            path=droot/'synthetic_models'/row['split']/(row['sample_id']+'.npz')
            if path.exists() and not args.overwrite:
                with np.load(path) as f:
                    if (str(f['case'].item())!=row['case'] or
                        int(f['case_index'])!=row['case_index'] or
                        int(f['noise_percent'])!=row['noise_percent']):
                        raise ValueError(f'Mismatching existing density model: {path}')
                continue
            density=make_density(row['case'],args.seed+SPLIT_OFFSET[row['split']],row['case_index'])
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,density=density,case=row['case'],
                                case_index=row['case_index'],noise_percent=row['noise_percent'],
                                split=row['split'],cell_size_m=10.0)
            if i%100==0 or i==len(rows):print(f'Models {i}/{len(rows)}',flush=True)
        manifest.parent.mkdir(parents=True,exist_ok=True)
        with manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    else:
        if not manifest.exists():raise FileNotFoundError(f'Generate models first: {manifest}')
        with manifest.open(newline='') as f:existing=list(csv.DictReader(f))
        if [(r['sample_id'],r['case'],int(r['noise_percent'])) for r in existing] != [
            (r['sample_id'],r['case'],r['noise_percent']) for r in rows]:
            raise ValueError('Manifest differs from requested seed/counts')
    if args.stage in ('physics','all'):
        jobs=[]
        for r in rows:
            fn=r['sample_id']+'.npz';split=r['split']
            jobs.append((str(droot/'synthetic_models'/split/fn),
                         str(droot/'forward_models'/split/fn),
                         str(proot/split/fn),args.seed,args.overwrite))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(physics_one,job) for job in jobs]
            for i,ft in enumerate(as_completed(futures),1):
                ft.result()
                if i%25==0 or i==len(jobs):print(f'Gzz forward+migration {i}/{len(jobs)}',flush=True)
    droot.mkdir(parents=True,exist_ok=True)
    (droot/'dataset_config.json').write_text(json.dumps(dict(
        experiment='E07_Gzz',forward_component='Gzz',unit='Eotvos',
        forward='MATLAB getPredGR case 7 (cell-centered)',
        migration='MATLAB gg_migration2014 migid 7 (depth-squared weighting)',
        sample_counts={'train':args.train,'validation':args.validation,
                       'test':args.test_per_case*len(CASES)*len(LEVELS)},
        seed=args.seed,geologies=list(CASES),noise_levels=list(LEVELS),
        noise_definition='Gaussian std as % peak absolute clean Gzz (E)'),indent=2))
    print('E07_Gzz dataset stage complete',flush=True)

if __name__=='__main__':main()
