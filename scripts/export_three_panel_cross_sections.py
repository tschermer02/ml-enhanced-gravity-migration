"""Export 2-D cross-section comparisons from saved prediction archives.

Columns: Synthetic Model | Migration | Inversion.
Rows: XY (constant z), XZ (constant y), YZ (constant x).
For multi-body models, exports one overview and a body-centered 3x3 figure
for EACH connected true-density body, with matching slice coordinates in
all three columns. Body-centered figures show full domains by default.

Examples:
  python scripts/export_three_panel_cross_sections.py --experiment E07/multiple/noise_05 --limit 2
  python scripts/export_three_panel_cross_sections.py --experiment E07_Gzz/salt_dome/noise_10 --limit 3
  python scripts/export_three_panel_cross_sections.py --all --limit 1
  python scripts/export_three_panel_cross_sections.py --experiment E06_signed/E06B --samples 1 4
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'outputs' / 'ml' / 'predict'
SHAPE = (24, 64, 64)
CELL_M = 10.0
DENSITY_THRESHOLD = 0.1
COLUMNS = ('Synthetic Model', 'Migration', 'Inversion')
ROWS = ('XY', 'XZ', 'YZ')


def _name(value: str) -> str:
    value = str(value).removesuffix('.npz')
    if value.startswith('sample_'):
        return value
    if value.isdigit():
        return f'sample_{int(value):06d}'
    return value


def _load(path: Path, migration_source: str):
    with np.load(path) as archive:
        required = ('density', 'prediction')
        for key in required:
            if key not in archive:
                raise KeyError(f'{path}: missing {key}; available keys: {archive.files}')
        density = np.asarray(archive['density'], dtype=np.float32).squeeze()
        inversion = np.asarray(archive['prediction'], dtype=np.float32).squeeze()
        if migration_source == 'normalized':
            if 'migration_input' in archive:
                migration = np.asarray(archive['migration_input'], dtype=np.float32).squeeze()
            elif 'migration' in archive:
                raw = np.asarray(archive['migration'], dtype=np.float32).squeeze()
                migration = raw / max(float(np.max(np.abs(raw))), 1e-8)
            else:
                raise KeyError(f'{path}: no migration_input or migration field')
        else:
            if 'migration' not in archive:
                raise KeyError(f'{path}: no raw migration field')
            migration = np.asarray(archive['migration'], dtype=np.float32).squeeze()
    for key, arr in [('density', density), ('migration', migration), ('prediction', inversion)]:
        if arr.shape != SHAPE or not np.isfinite(arr).all():
            raise ValueError(f'{path}: {key} has invalid shape/values {arr.shape}')
    return density, migration, inversion


def _components(density, threshold, connectivity, min_body_voxels):
    mask = np.abs(density) >= threshold
    struct = ndimage.generate_binary_structure(3, connectivity)
    labeled, n = ndimage.label(mask, structure=struct)
    bodies = []
    for identifier in range(1, n+1):
        indices = np.argwhere(labeled == identifier)
        if len(indices) < min_body_voxels:
            continue
        centroid = np.mean(indices, axis=0)
        center = tuple(int(np.clip(np.round(v), 0, d-1)) for v,d in zip(centroid, SHAPE))
        bodies.append((len(indices), center, identifier))
    bodies.sort(key=lambda item: (-item[0], item[1]))
    return bodies


def _slices(volume, center):
    z, y, x = center
    return [volume[z, :, :], volume[:, y, :], volume[:, :, x]]


def _figure(truth, mig, pred, center, title, output, migration_source, dpi):
    volumes = [truth, mig, pred]
    nz, ny, nx = truth.shape
    density_max = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 0.1)
    density_norm = TwoSlopeNorm(vmin=-density_max, vcenter=0, vmax=density_max)
    migmax = max(float(np.max(np.abs(mig))), 1e-6)
    mig_norm = TwoSlopeNorm(vmin=-migmax, vcenter=0, vmax=migmax)
    fig, axes = plt.subplots(3, 3, figsize=(14.4, 11.2), constrained_layout=True,
                             gridspec_kw={'height_ratios':[1.0, 0.78, 0.78]})
    for col, volume in enumerate(volumes):
        for row, section in enumerate(_slices(volume, center)):
            ax = axes[row, col]
            # z=0 at top for vertical slices, x/y show geographic increasing coordinates.
            if row == 0:
                extent = [0, nx*CELL_M, 0, ny*CELL_M]
                origin = 'lower'
                xlabel, ylabel = 'East (m)', 'North (m)'
            elif row == 1:
                extent = [0, nx*CELL_M, nz*CELL_M, 0]
                origin = 'upper'
                xlabel, ylabel = 'East (m)', 'Depth (m)'
            else:
                extent = [0, ny*CELL_M, nz*CELL_M, 0]
                origin = 'upper'
                xlabel, ylabel = 'North (m)', 'Depth (m)'
            im = ax.imshow(section, cmap='RdBu_r',
                           norm=mig_norm if col == 1 else density_norm,
                           origin=origin, extent=extent, interpolation='nearest',
                           aspect='auto')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            if col == 0:
                z,y,x=center
                ax.text(0.02, 0.97, [f'XY · depth {(z+0.5)*CELL_M:.0f} m',
                                      f'XZ · north {(y+0.5)*CELL_M:.0f} m',
                                      f'YZ · east {(x+0.5)*CELL_M:.0f} m'][row],
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(facecolor='white',alpha=.82,edgecolor='none',pad=3))
            if row == 0: ax.set_title(COLUMNS[col], fontsize=14, weight='bold', pad=12)
    bar_truth = fig.colorbar(axes[0, 0].images[0], ax=[axes[r,c] for c in (0,2) for r in range(3)],
                             orientation='horizontal', shrink=.72, pad=.08, aspect=38)
    bar_truth.set_label('Density contrast (g/cm³) · shared true/pred scale', fontsize=10)
    bar_mig = fig.colorbar(axes[0, 1].images[0], ax=[axes[r,1] for r in range(3)],
                           orientation='horizontal', shrink=.83, pad=.08, aspect=18)
    bar_mig.set_label('Normalized migration (unitless)' if migration_source == 'normalized'
                      else 'Migration amplitude (not calibrated density)', fontsize=10)
    fig.suptitle(title, fontsize=16, weight='bold')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output


def export_one(path, prediction_root, split, threshold, connectivity,
               min_body_voxels, migration_source, dpi, overwrite, per_body):
    truth, migration, prediction = _load(path, migration_source)
    bodies = _components(truth, threshold, connectivity, min_body_voxels)
    if not bodies:
        raise ValueError(f'{path}: no true bodies at threshold {threshold}')
    relative = path.relative_to(prediction_root)
    # predictions/test/<sample>.npz -> figures/test/<sample>_cross_sections.png
    parts = relative.parts
    if len(parts) < 3 or parts[-3:-1] != ('predictions', split):
        raise ValueError(f'Unexpected prediction layout: {path}')
    experiment_parts = parts[:-3]
    outdir = prediction_root.joinpath(*experiment_parts, 'figures', split)
    center = tuple(int(np.round(v)) for v in np.mean(np.argwhere(np.abs(truth)>=threshold),axis=0))
    outputs = []
    overview = outdir / f'{path.stem}_cross_sections.png'
    if overwrite or not overview.exists():
        outputs.append(_figure(truth,migration,prediction,center,
                    f'{path.stem} | Full-model cross-sections', overview,migration_source,dpi))
    if per_body:
        for number, (count, body_center, _) in enumerate(bodies, 1):
            output = outdir / f'{path.stem}_body_{number:02d}_cross_sections.png'
            if overwrite or not output.exists():
                outputs.append(_figure(truth,migration,prediction,body_center,
                    f'{path.stem} | Body {number} ({count} voxels) — matched cross-sections',
                    output,migration_source,dpi))
    return outputs, len(bodies)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    selector=p.add_mutually_exclusive_group(required=True)
    selector.add_argument('--experiment',help='e.g. E07/multiple/noise_05')
    selector.add_argument('--all',action='store_true',help='Find every predictions/test folder')
    p.add_argument('--predictions-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--split',choices=('test','validation'),default='test')
    p.add_argument('--samples',nargs='+',help='sample IDs (1, sample_000001, multiple_000001_n05)')
    p.add_argument('--limit',type=int,default=None,help='limit number of samples PER experiment')
    p.add_argument('--threshold',type=float,default=DENSITY_THRESHOLD)
    p.add_argument('--connectivity',type=int,choices=(1,2,3),default=1)
    p.add_argument('--min-body-voxels',type=int,default=8)
    p.add_argument('--migration-source',choices=('normalized','raw'),default='normalized')
    p.add_argument('--dpi',type=int,default=180)
    p.add_argument('--no-per-body',action='store_true',help='overview only')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    if a.threshold<=0 or a.min_body_voxels<1 or a.dpi<30:p.error('invalid threshold / min-body-voxels / dpi')
    root=a.predictions_root if a.predictions_root.is_absolute() else ROOT/a.predictions_root
    dirs=sorted(root.glob(f'**/predictions/{a.split}')) if a.all else [root/a.experiment/'predictions'/a.split]
    if not any(d.is_dir() for d in dirs):
        raise FileNotFoundError(f'No prediction directories found under {root}')
    total=0
    wanted={_name(s) for s in a.samples} if a.samples else None
    for folder in dirs:
        if not folder.is_dir():
            print('Missing:',folder,flush=True);continue
        files=sorted(folder.glob('*.npz'))
        if wanted is not None:files=[f for f in files if f.stem in wanted]
        if a.limit is not None:files=files[:a.limit]
        for file in files:
            outputs,nbody=export_one(file,root,a.split,a.threshold,a.connectivity,
                       a.min_body_voxels,a.migration_source,a.dpi,a.overwrite,not a.no_per_body)
            print(f'{file}: {nbody} true bodies, {len(outputs)} new figure(s)',flush=True)
            for output in outputs:print('  saved:',output,flush=True)
            total+=len(outputs)
    print(f'Completed: {total} PNG figure(s).',flush=True)

if __name__=='__main__':main()
