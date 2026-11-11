"""Save E10A/E10B voxel predictions for synthetic held-out test samples.

Run from repository root:
 python -m src.prediction.predict_e10 --name E10A \
   --data-root /scratch/general/vast/$USER/e10/data \
   --model /scratch/general/vast/$USER/e10/E10A/checkpoints/best_unet3d.keras \
   --output-root outputs/ml/predict/E10A

Uses exactly the training-time input transformation: E10A per-sample
max-absolute normalization; E10B unscaled raw migration.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

SHAPE = (24, 160, 160)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', choices=('E10A', 'E10B'), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    root = args.data_root.expanduser().resolve()
    source = root / args.split
    paths = sorted(source.rglob('*.npz'))
    if args.limit is not None:
        if args.limit < 1: p.error('--limit must be positive')
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f'No test samples found under {source}')
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    out = args.output_root.expanduser().resolve()
    targets = [out / args.split / path.relative_to(source) for path in paths]
    present = [str(t) for t in targets if t.exists()]
    if present and not args.overwrite:
        raise FileExistsError(f'{len(present)} predictions already exist, e.g. {present[0]} (use --overwrite)')
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model, compile=False)
    expected_shape = (None, *SHAPE, 1)
    if tuple(model.input_shape) != expected_shape:
        raise ValueError(f'Model input {model.input_shape} != {expected_shape}')

    manifest = []
    for i, (path, dest) in enumerate(zip(paths, targets), 1):
        with np.load(path, allow_pickle=False) as f:
            migration = np.asarray(f['migration'], np.float32)
            density = np.asarray(f['density'], np.float32)
            gravity = np.asarray(f['gravity'], np.float32) if 'gravity' in f else None
            geometry = str(f['geometry']) if 'geometry' in f else path.parent.name
            noise = float(f['noise_percent']) if 'noise_percent' in f else float('nan')
        if migration.shape != SHAPE or density.shape != SHAPE:
            raise ValueError(f'{path}: unexpected migration/density shape')
        scale = max(float(np.max(np.abs(migration))), 1e-8)
        network_input = (migration / scale if args.name == 'E10A' else migration).astype(np.float32)
        if not np.isfinite(network_input).all():
            raise ValueError(f'{path}: non-finite network input')
        y = np.asarray(model(network_input[None, ..., None], training=False))[0, ..., 0]
        if y.shape != SHAPE or not np.isfinite(y).all():
            raise ValueError(f'{path}: invalid prediction {y.shape}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(prediction=y.astype(np.float32), density=density,
                       migration=migration, migration_input=network_input,
                       migration_scale=np.float64(scale), geometry=np.str_(geometry),
                       noise_percent=np.float32(noise))
        if gravity is not None: payload['gravity'] = gravity
        np.savez_compressed(dest, **payload)
        manifest.append(dict(source=str(path), output=str(dest), geometry=geometry,
                             noise_percent=noise))
        if i % 10 == 0 or i == len(paths):
            print(f'{args.name}: {i}/{len(paths)} predictions saved', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'{args.split}_prediction_manifest.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=manifest[0].keys())
        writer.writeheader(); writer.writerows(manifest)
    (out / f'{args.split}_prediction_config.json').write_text(json.dumps({
        'name': args.name, 'split': args.split, 'count': len(paths),
        'dataset': str(root), 'checkpoint': str(args.model.resolve()),
        'network_input': 'maxabs normalized' if args.name == 'E10A' else 'raw migration',
        'volume_shape': SHAPE
    }, indent=2))
    print('Predictions written to:', out, flush=True)


if __name__ == '__main__':
    main()
