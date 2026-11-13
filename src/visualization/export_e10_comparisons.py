"""E10A/E10B figures in the established E08 three-column, XY/XZ/YZ style.

For each saved E10 prediction, export:
  * 3x3 density comparison (true density | migration | inversion), overview
    and optionally each connected true body, all sliced through actual bodies
  * three-panel Gzz maps (observed | calculated | residual)

Usage (from repository root):
 python -m src.visualization.export_e10_comparisons --name E10A --limit 5
 python -m src.visualization.export_e10_comparisons --name E10B --type density
 python -m src.visualization.export_e10_comparisons --name E10A --type gravity --limit 5

E10 domain: (z,y,x)=(24,160,160), spacings (25,50,50) m.
Inputs: outputs/ml/predict/E10{A,B}/test/<geometry>/*.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy import ndimage

from src.data.e10_regional import forward_gzz, SHAPE, OBS_SHAPE, DX, DY, DZ

ROOT = Path(__file__).resolve().parents[2]
COLS = ('Synthetic Model', 'Migration', 'Inversion')


def load_sample(path: Path, migration_source: str):
    with np.load(path, allow_pickle=False) as f:
        missing = {'density', 'prediction', 'migration'} - set(f.files)
        if missing:
            raise KeyError(f'{path}: missing {sorted(missing)}')
        truth = np.asarray(f['density'], dtype=np.float32).squeeze()
        pred = np.asarray(f['prediction'], dtype=np.float32).squeeze()
        raw = np.asarray(f['migration'], dtype=np.float32).squeeze()
        if migration_source == 'normalized':
            mig = raw / max(float(np.max(np.abs(raw))), 1e-8)
        else:
            mig = raw
        obs = np.asarray(f['gravity'], dtype=np.float32).squeeze() if 'gravity' in f else None
    for name, arr in [('density', truth), ('prediction', pred), ('migration', mig)]:
        if arr.shape != SHAPE or not np.isfinite(arr).all():
            raise ValueError(f'{path}: {name} invalid {arr.shape} (expected {SHAPE})')
    if obs is not None and (obs.shape != OBS_SHAPE or not np.isfinite(obs).all()):
        raise ValueError(f'{path}: gravity invalid {obs.shape} (expected {OBS_SHAPE})')
    return truth, mig, pred, obs


def body_components(truth, threshold: float, connectivity: int, min_voxels: int):
    mask = np.abs(truth) >= threshold
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, connectivity))
    found = []
    for idx in range(1, count + 1):
        occupied = np.argwhere(labels == idx)
        if len(occupied) < min_voxels:
            continue
        center = np.rint(occupied.mean(axis=0)).astype(int)
        # A geometric centroid can fall outside a concave body: ensure slice
        # coordinates actually traverse an occupied voxel.
        nearest = np.argmin(np.sum((occupied - center[None, :])**2, axis=1))
        center = tuple(int(v) for v in occupied[nearest])
        found.append((len(occupied), center))
    found.sort(key=lambda x: -x[0])
    return found


def overview_center(truth, threshold):
    occupied = np.argwhere(np.abs(truth) >= threshold)
    if not len(occupied):
        raise ValueError('No true density voxels above threshold; check dataset or reduce --threshold')
    center = occupied.mean(axis=0)
    nearest = np.argmin(np.sum((occupied - center[None, :])**2, axis=1))
    return tuple(int(i) for i in occupied[nearest])


def plot_density(truth, mig, pred, center, destination, title, *, source='raw', dpi=180):
    z, y, x = center
    nz, ny, nx = truth.shape
    extent = [
        (0, nx*DX, 0, ny*DY),
        (0, nx*DX, nz*DZ, 0),
        (0, ny*DY, nz*DZ, 0),
    ]
    slices = lambda vol: (vol[z,:,:], vol[:,y,:], vol[:,:,x])
    section_labels = (
        f'XY · depth {(z+0.5)*DZ:.0f} m',
        f'XZ · north {(y+0.5)*DY:.0f} m',
        f'YZ · east {(x+0.5)*DX:.0f} m',
    )
    axes_labels = (('East (m)', 'North (m)'), ('East (m)', 'Depth (m)'), ('North (m)', 'Depth (m)'))
    density_max = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 1e-6)
    mig_max = max(float(np.max(np.abs(mig))), 1e-6)
    density_norm = TwoSlopeNorm(vmin=-density_max, vcenter=0, vmax=density_max)
    migration_norm = TwoSlopeNorm(vmin=-mig_max, vcenter=0, vmax=mig_max)

    fig, axs = plt.subplots(3, 3, figsize=(17, 13), gridspec_kw={'height_ratios': [1, .78, .78]})
    fig.subplots_adjust(left=.08, right=.98, top=.90, bottom=.20, wspace=.28, hspace=.43)
    images = {}
    for col, vol in enumerate((truth, mig, pred)):
        for row, section in enumerate(slices(vol)):
            ax = axs[row,col]
            im = ax.imshow(section, cmap='RdBu_r', origin='lower' if row==0 else 'upper',
                           extent=extent[row], aspect='auto', interpolation='nearest',
                           norm=migration_norm if col==1 else density_norm)
            images[col] = im
            ax.set_xlabel(axes_labels[row][0], fontsize=13)
            ax.set_ylabel(axes_labels[row][1], fontsize=13)
            ax.tick_params(axis='both', labelsize=11)
            if col == 0:
                ax.text(.02, .97, section_labels[row], transform=ax.transAxes,
                        va='top', fontsize=12,
                        bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=4))
            if row == 0:
                ax.set_title(COLS[col], fontsize=17, weight='bold', pad=14)

    cax1 = fig.add_axes([.12,.095,.35,.018])
    cbar = fig.colorbar(images[0], cax=cax1, orientation='horizontal')
    cbar.set_label(r'Density contrast (g/cm$^3$)', fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=11)
    cax2 = fig.add_axes([.60,.095,.32,.018])
    cbar = fig.colorbar(images[1], cax=cax2, orientation='horizontal')
    cbar.set_label('Normalized migration (unitless)' if source=='normalized'
                   else 'Raw migration (not calibrated density)', fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=11)
    fig.suptitle(title, fontsize=19, weight='bold', y=.97)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=max(dpi, 120), bbox_inches='tight')
    plt.close(fig)


def plot_gravity(obs, prediction, destination, title, *, dpi=180):
    if obs is None:
        raise ValueError('Prediction archive does not contain observed gravity')
    calculated = forward_gzz(prediction)
    residual = obs - calculated
    rmse = float(np.sqrt(np.mean(residual.astype(np.float64)**2)))
    scale = max(float(np.max(np.abs(obs))), float(np.max(np.abs(calculated))), 1e-6)
    rscale = max(float(np.max(np.abs(residual))), 1e-6)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.subplots_adjust(left=.06, right=.97, top=.83, bottom=.17, wspace=.31)
    vals = (obs, calculated, residual)
    names = ('Observed Gzz', 'Forward Gzz from prediction', 'Residual (observed − predicted)')
    for ax, val, name, lim in zip(axs, vals, names, (scale, scale, rscale)):
        im = ax.imshow(val, origin='lower', cmap='RdBu_r', aspect='equal',
                       extent=(0, (OBS_SHAPE[1]-1)*DX, 0, (OBS_SHAPE[0]-1)*DY),
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
        ax.set(title=name, xlabel='East (m)', ylabel='North (m)')
        fig.colorbar(im, ax=ax, shrink=.82, label='Gzz (Eötvös)')
    fig.suptitle(f'{title} | Gzz RMSE = {rmse:.4g} E', fontsize=15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return rmse


def selected_paths(base, names, limit):
    paths = sorted(base.rglob('*.npz'))
    if names:
        wanted = set()
        for item in names:
            stem = str(item).removesuffix('.npz').removeprefix('sample_')
            wanted.add(f'sample_{int(stem):06d}' if stem.isdigit() else str(item))
        paths = [p for p in paths if p.stem in wanted]
    return paths[:limit] if limit is not None else paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', choices=('E10A','E10B'), required=True)
    p.add_argument('--predictions-root', type=Path, default=ROOT/'outputs/ml/predict')
    p.add_argument('--split', choices=('test','validation','train'), default='test')
    p.add_argument('--type', choices=('both','density','gravity'), default='both')
    p.add_argument('--samples', nargs='+', default=None)
    p.add_argument('--limit', type=int)
    p.add_argument('--threshold', type=float, default=.1)
    p.add_argument('--connectivity', type=int, choices=(1,2,3), default=1)
    p.add_argument('--min-body-voxels', type=int, default=8)
    p.add_argument('--migration-source', choices=('normalized','raw'), default='raw')
    p.add_argument('--no-per-body', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dpi', type=int, default=180)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    if args.threshold <= 0 or args.min_body_voxels < 1: p.error('--threshold and --min-body-voxels must be positive')
    root = args.predictions_root.expanduser().resolve()
    base = root / args.name / args.split
    paths = selected_paths(base, args.samples, args.limit)
    if not paths: raise FileNotFoundError(f'No prediction NPZ files found under {base}')
    for i,path in enumerate(paths,1):
        relative = path.relative_to(base)
        figdir = root / args.name / 'figures' / args.split / relative.parent
        truth, mig, pred, obs = load_sample(path, args.migration_source)
        if args.type in ('both','density'):
            center = overview_center(truth, args.threshold)
            destination = figdir / f'{path.stem}_cross_sections.png'
            if args.overwrite or not destination.exists():
                plot_density(truth,mig,pred,center,destination,
                             f'{path.stem} | Full-model cross-sections',source=args.migration_source,dpi=args.dpi)
            if not args.no_per_body:
                bodies = body_components(truth,args.threshold,args.connectivity,args.min_body_voxels)
                for number,(count,bc) in enumerate(bodies,1):
                    dest = figdir/f'{path.stem}_body_{number:02d}_cross_sections.png'
                    if args.overwrite or not dest.exists():
                        plot_density(truth,mig,pred,bc,dest,
                                     f'{path.stem} | Body {number} ({count} voxels) — matched cross-sections',
                                     source=args.migration_source,dpi=args.dpi)
        if args.type in ('both','gravity'):
            dest = figdir/f'{path.stem}_gravity_comparison.png'
            if args.overwrite or not dest.exists():
                rmse = plot_gravity(obs,pred,dest,path.stem,dpi=args.dpi)
                print(f'{args.name}: {i}/{len(paths)} {path.stem} | Gzz RMSE={rmse:.4g} E', flush=True)
        print(f'{args.name}: {i}/{len(paths)} {path.stem} -> {figdir}',flush=True)

if __name__=='__main__':
    main()
