"""Export observed, predicted, and residual surface gravity maps from saved predictions.

Run at repository root:
    python -m src.visualization.export_gravity_comparison --all
    python -m src.visualization.export_gravity_comparison --name E08A --limit 3

Input NPZ must contain 'prediction' and 'gravity'. Component is inferred from
'observation_component' (saved by generator); optional --component override.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from src.physics.gravity_forward import calculate_gravity

ROOT = Path(__file__).resolve().parents[2]


def export_one(path: Path, output: Path, component: str | None = None, dpi: int = 170):
    with np.load(path, allow_pickle=False) as sample:
        required = {'prediction', 'gravity'}
        if not required.issubset(sample.files):
            raise KeyError(f'{path}: missing keys {required - set(sample.files)}')
        prediction = np.asarray(sample['prediction'], dtype=np.float64).squeeze()
        observed = np.asarray(sample['gravity'], dtype=np.float64).squeeze()
        if component is None:
            if 'observation_component' not in sample:
                raise ValueError(f'{path}: no observation_component; use --component gz or gzz')
            component = str(sample['observation_component'].item()).lower()
    if component not in ('gz', 'gzz'):
        raise ValueError(f'Invalid gravity component: {component}')
    if prediction.shape != (24, 64, 64) or observed.shape != (81, 81):
        raise ValueError(f'{path}: unexpected shapes: {prediction.shape}, {observed.shape}')
    if not np.isfinite(prediction).all() or not np.isfinite(observed).all():
        raise ValueError(f'{path}: nonfinite prediction or gravity')

    predicted = np.asarray(calculate_gravity(prediction, component=component), dtype=np.float64)
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    unit = 'mGal' if component == 'gz' else 'E'
    scale = max(float(np.max(np.abs(observed))), float(np.max(np.abs(predicted))), 1e-12)
    rscale = max(float(np.max(np.abs(residual))), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for i, (values, title) in enumerate(((observed, 'Observed gravity'),
                                         (predicted, 'Predicted-density forward gravity'),
                                         (residual, 'Residual (observed − predicted)'))):
        norm = TwoSlopeNorm(vmin=-(rscale if i == 2 else scale), vcenter=0,
                            vmax=(rscale if i == 2 else scale))
        im = axes[i].imshow(values, origin='lower', cmap='RdBu_r', norm=norm,
                             extent=[-85, 715, -85, 715])
        axes[i].set(title=title, xlabel='Easting (m)', ylabel='Northing (m)')
        fig.colorbar(im, ax=axes[i], label=f'{component.upper()} ({unit})', shrink=0.8)
    fig.suptitle(f'{path.stem} | {component.upper()} gravity comparison | RMSE = {rmse:.4g} {unit}')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--name', help='Experiment (e.g. E08A)')
    selector.add_argument('--all', action='store_true')
    parser.add_argument('--split', default='test', choices=('test', 'validation', 'train'))
    parser.add_argument('--predictions-root', type=Path, default=ROOT / 'outputs/ml/predict')
    parser.add_argument('--component', choices=('gz', 'gzz'), default=None,
                        help='Override component if older archives omit metadata')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dpi', type=int, default=170)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    base = args.predictions_root
    if not base.is_absolute():
        base = ROOT / base
    folders = sorted(base.glob(f'**/predictions/{args.split}')) if args.all else [base / args.name / 'predictions' / args.split]
    if not any(p.is_dir() for p in folders):
        raise FileNotFoundError(f'No prediction folders for {args.split} under {base}')
    total = 0
    for folder in folders:
        paths = sorted(folder.glob('*.npz'))
        if args.limit is not None:
            paths = paths[:args.limit]
        for i, path in enumerate(paths, 1):
            destination = folder.parent.parent / 'figures' / args.split / f'{path.stem}_gravity_comparison.png'
            if destination.exists() and not args.overwrite:
                continue
            rmse = export_one(path, destination, args.component, args.dpi)
            total += 1
            print(f'{folder.parent.parent.name}: {i}/{len(paths)} | RMSE {rmse:.4g} | {destination}', flush=True)
    print(f'Saved {total} gravity comparison figure(s)', flush=True)


if __name__ == '__main__':
    main()
