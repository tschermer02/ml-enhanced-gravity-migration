"""Export every test density model to MATLAB-compatible HDF5 and index.csv."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np


def test_samples(dataset: Path):
    # Supports a dataset root or a split directory. Prefer validation when present,
    # so the default workflow runs the full 100-sample validation set.
    manifest = dataset / 'test_manifest.csv'
    if manifest.exists():
        with manifest.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            path = dataset / row['relative_path']
            yield row.get('sample_id', path.stem), path
        return

    for split_name in ('validation', 'val', 'test'):
        split_dir = dataset / split_name
        if split_dir.exists():
            for path in sorted(split_dir.glob('*.npz')):
                yield path.stem, path
            return

    for path in sorted(dataset.glob('*.npz')):
        yield path.stem, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('data/synthetic/single_prism'))
    p.add_argument('--output', type=Path, default=Path('analysis_outputs/gravity_forward_validation/inputs'))
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()
    samples = list(test_samples(args.dataset))
    if args.limit is not None:
        if args.limit < 1: p.error('--limit must be >= 1')
        samples = samples[:args.limit]
    if not samples: raise FileNotFoundError(f'No test NPZ files found in {args.dataset}')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (sid, source) in enumerate(samples, 1):
        with np.load(source) as arrays:
            density = np.asarray(arrays['density'], dtype=np.float64)
            saved_gravity = np.asarray(arrays['gravity'], dtype=np.float64)
        if density.shape != (24, 64, 64) or saved_gravity.shape != (81, 81):
            raise ValueError(f'{sid}: wrong shape {density.shape}, {saved_gravity.shape}')
        destination = args.output / f'{sid}.h5'
        with h5py.File(destination, 'w') as f:
            # C-flatten (z,y,x) is x-fastest and matches MATLAB reshape([64,64,24]).
            f.create_dataset('density_xfast_gcm3', data=density.ravel(order='C'))
            f.create_dataset('stored_gz_xfast_mgal', data=saved_gravity.ravel(order='C'))
        rows.append({'sample_id': sid, 'source': str(source), 'h5_name': destination.name})
        print(f'Exported {i}/{len(samples)}: {sid}', flush=True)
    with (args.output / 'index.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'source', 'h5_name'])
        writer.writeheader(); writer.writerows(rows)
    (args.output / 'metadata.json').write_text(json.dumps({'sample_count':len(rows),'density_shape_zyx':[24,64,64],'gravity_shape_yx':[81,81],'cell_m':10,'receivers_x_y_m':'-85:10:715','receivers_z_m':0,'density_gcm3':True,'gravity_mgal':True,'flattening':'x fastest, then y, then z','method':'cell-centered point mass; G=6.67e-11'},indent=2))
    print(f'Exported {len(rows)} test cases to {args.output}')

if __name__ == '__main__': main()
