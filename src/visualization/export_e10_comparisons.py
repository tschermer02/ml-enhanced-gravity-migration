"""E10 synthetic three-panel density sections and observed/predicted/residual Gzz.

Run from repository root:
python -m src.visualization.export_e10_comparisons --name E10A --limit 5
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from src.data.e10_regional import forward_gzz, SHAPE, OBS_SHAPE, DX, DY, DZ

ROOT = Path(__file__).resolve().parents[2]

def plot_section(ax, arr, axis, index, title, scale, unit):
    # Cross section XZ through the center of the Y axis, depth downward.
    section = arr[:,index,:]
    im = ax.imshow(section, origin='upper', extent=(0,SHAPE[2]*DX, SHAPE[0]*DZ,0),
                   cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale), aspect='auto')
    ax.set(title=title, xlabel='Easting within model (m)', ylabel='Depth (m)')
    ax.figure.colorbar(im, ax=ax, shrink=.78, label=unit)

def create_figures(path:Path, folder:Path, dpi:int):
    with np.load(path, allow_pickle=False) as f:
        required = {'density','migration','prediction','gravity'}
        if not required.issubset(f.files):
            raise ValueError(f'{path}: missing {sorted(required-set(f.files))}')
        true = np.asarray(f['density'],dtype=np.float32)
        mig = np.asarray(f['migration'],dtype=np.float32)
        pred = np.asarray(f['prediction'],dtype=np.float32)
        obs = np.asarray(f['gravity'],dtype=np.float32)
    for name, val, shape in [('density',true,SHAPE),('migration',mig,SHAPE),('prediction',pred,SHAPE),('gravity',obs,OBS_SHAPE)]:
        if val.shape != shape or not np.isfinite(val).all():
            raise ValueError(f'{path}: {name} invalid shape {val.shape} or nonfinite values')
    folder.mkdir(parents=True,exist_ok=True)
    idx = SHAPE[1]//2
    fig, axs = plt.subplots(1,3,figsize=(18,5.3),constrained_layout=True)
    density_scale = max(float(np.max(np.abs(true))),float(np.max(np.abs(pred))),1e-9)
    mig_scale = max(float(np.max(np.abs(mig))),1e-9)
    plot_section(axs[0],true,1,idx,'True density',density_scale,'g/cm³')
    plot_section(axs[1],mig,1,idx,'Gzz migration (raw)',mig_scale,'migration units')
    plot_section(axs[2],pred,1,idx,'Predicted density',density_scale,'g/cm³')
    fig.suptitle(f'{path.parent.name} / {path.stem}: Y = {(idx+.5)*DY:.0f} m')
    fig.savefig(folder/f'{path.stem}_three_panel_density.png',dpi=dpi)
    plt.close(fig)

    calc = forward_gzz(pred)
    residual = obs-calc
    rmse = float(np.sqrt(np.mean(residual**2)))
    data_scale = max(float(np.max(np.abs(obs))),float(np.max(np.abs(calc))),1e-9)
    res_scale = max(float(np.max(np.abs(residual))),1e-9)
    fig, axs = plt.subplots(1,3,figsize=(18,5.3),constrained_layout=True)
    for ax, dat, title, scale in zip(axs,(obs,calc,residual),('Observed Gzz','Forward Gzz from prediction','Residual (observed − predicted)'),(data_scale,data_scale,res_scale)):
        im=ax.imshow(dat,origin='lower',extent=(0,OBS_SHAPE[1]*DX,0,OBS_SHAPE[0]*DY),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale))
        ax.set(title=title,xlabel='Easting (m)',ylabel='Northing (m)')
        fig.colorbar(im,ax=ax,shrink=.78,label='Eötvös')
    fig.suptitle(f'{path.stem} | Gzz RMSE = {rmse:.4g} E')
    fig.savefig(folder/f'{path.stem}_gravity_comparison.png',dpi=dpi)
    plt.close(fig)
    return rmse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',choices=['E10A','E10B'],required=True)
    p.add_argument('--predictions-root',type=Path,default=ROOT/'outputs/ml/predict')
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--dpi',type=int,default=170)
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be >0')
    base=args.predictions_root/args.name/'test'
    paths=sorted(base.rglob('*.npz'))
    if args.limit:paths=paths[:args.limit]
    if not paths:raise FileNotFoundError(f'No predictions under {base}. Run predict_e10.py first.')
    for i,path in enumerate(paths,1):
        destination=args.predictions_root/args.name/'figures'/path.parent.name
        rmse=create_figures(path,destination,args.dpi)
        print(f'{args.name}: {i}/{len(paths)} {path.name} | Gzz RMSE={rmse:.4g} E | {destination}',flush=True)

if __name__=='__main__': main()
