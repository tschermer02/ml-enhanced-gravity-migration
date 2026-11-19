"""E12 Thunderbird: preserve actual survey XY and Gzz; strict coverage checks.

Prepare 81x81 survey observation map, coverage, nearest-station distance.
Optionally infer ONLY when all 6561 samples are within survey convex hull.
No rescaling, nearest-neighbor extrapolation, migration, or inversion by default.

  python -m src.E12.field --prepare-only
  python -m src.E12.field --experiment E12F
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from .config import EXPERIMENTS,OBS_SHAPE,EXTENT_M,FLIGHT_HEIGHT_M
from .physics import forward,migration,normalize_migration
ROOT=Path(__file__).resolve().parents[2]

def survey(path,channel=7):
    a=np.loadtxt(path)
    if a.ndim!=2 or a.shape[1]!=5 or not np.isfinite(a).all():raise ValueError('Expected five finite columns')
    a=a[a[:,3]==channel]
    if len(a)<100:raise ValueError('Missing AGG component')
    xy=a[:,:2]; vals=a[:,4]
    unique,inverse,counts=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
    if len(unique)!=len(xy):
        vals=np.bincount(inverse,weights=vals)/counts
        xy=unique
    return xy,vals,a[:,2]

def prepare(xy,values):
    bounds=np.array([xy.min(0),xy.max(0)])
    x=np.linspace(bounds[0,0],bounds[1,0],81)
    y=np.linspace(bounds[0,1],bounds[1,1],81)
    X,Y=np.meshgrid(x,y)
    targets=np.column_stack([X.ravel(),Y.ravel()])
    interpolated=LinearNDInterpolator(xy,values,fill_value=np.nan)(targets).reshape(OBS_SHAPE)
    valid=np.isfinite(interpolated)
    distance=cKDTree(xy).query(targets)[0].reshape(OBS_SHAPE)
    return dict(gravity=interpolated.astype(np.float32),
                coverage_mask=valid,nearest_station_distance_m=distance.astype(np.float32),
                x_coordinates_m=x,y_coordinates_m=y,
                bounds_m=bounds,
                actual_dx_m=float(x[1]-x[0]),actual_dy_m=float(y[1]-y[0]))

def map_plots(src,xy,values,out):
    X,Y=np.meshgrid(src['x_coordinates_m']/1000,src['y_coordinates_m']/1000)
    limit=max(float(np.percentile(np.abs(values),99)),1.e-6)
    fig,ax=plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
    a=ax[0].scatter(xy[:,0]/1000,xy[:,1]/1000,c=values,s=2,vmin=-limit,vmax=limit,cmap='RdBu_r',rasterized=True)
    b=ax[1].pcolormesh(X,Y,np.ma.masked_invalid(src['gravity']),vmin=-limit,vmax=limit,cmap='RdBu_r',shading='nearest')
    for axy,title in zip(ax,('Original station measurements','81×81 linear resampling (no extrapolation)')):
        axy.set(title=title,xlabel='X (km)',ylabel='Y (km)');axy.set_aspect('equal')
    fig.colorbar(a,ax=ax[0],label='Gzz (assumed Eotvos)')
    fig.colorbar(b,ax=ax[1],label='Gzz (assumed Eotvos)')
    fig.savefig(out/'observed_Gzz_original_vs_81.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    ax[0].pcolormesh(X,Y,src['coverage_mask'].astype(float),vmin=0,vmax=1,cmap='Greens',shading='nearest')
    im=ax[1].pcolormesh(X,Y,np.ma.masked_where(~src['coverage_mask'],src['nearest_station_distance_m']),cmap='viridis',shading='nearest')
    for axy,title in zip(ax,('Inside survey convex hull','Nearest measured station distance')):
        axy.set(title=title,xlabel='X (km)',ylabel='Y (km)');axy.set_aspect('equal')
    fig.colorbar(im,ax=ax[1],label='Meters')
    fig.savefig(out/'interpolation_coverage_map.png',dpi=220);plt.close(fig)

def infer(g,exp,height,out):
    import tensorflow as tf
    from .visualize import output_figures
    if not g['coverage_mask'].all():
        count=np.count_nonzero(~g['coverage_mask'])
        raise RuntimeError(f'{count} receiver nodes outside convex hull; refusing to extrapolate. '
                           'Field preprocessing and coverage maps are saved; no inference run. '
                           'Do not replace NaNs with zeros or nearest values without explicit study design.')
    dx=float(g['actual_dx_m']);dy=float(g['actual_dy_m'])
    if abs(dx-100)>1 or abs(dy-100)>1:
        raise RuntimeError(f'Actual receiver spacings ({dx:.2f}, {dy:.2f}) m differ from E12 (100,100) '
                           'by >1m; inference would distort physical scale; refusing')
    p=ROOT/'outputs/ml/training'/exp/'checkpoints/best_unet3d.keras'
    model=tf.keras.models.load_model(p,compile=False)
    mig=migration(g['gravity'],height)
    normalized,scale=normalize_migration(mig)
    pred=model(normalized[None,...,None],training=False).numpy()[0,...,0]
    modeled=forward(pred,height)
    res=modeled-g['gravity']
    np.savez_compressed(out/'Thunderbird_E12_prediction.npz',density=pred,
         migration=mig,migration_input=normalized,migration_scale=scale,
         modeled_gravity=modeled,residual=res,observed=g['gravity'],
         x_coordinates_m=g['x_coordinates_m'],y_coordinates_m=g['y_coordinates_m'],
         observation_height_m=height)
    output_figures(out,g['gravity'],modeled,mig,normalized,pred)
    print('Observed vs forward modeled RMSE E:',np.sqrt(np.mean(res**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('src/data/Thunderbird_gravity_data/Tbird_gravity.dat'))
    p.add_argument('--experiment',choices=EXPERIMENTS,default=None)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--height',type=float,default=FLIGHT_HEIGHT_M)
    p.add_argument('--channel',type=int,default=7)
    p.add_argument('--output-root',type=Path,default=Path('outputs/ml/field/Thunderbird_E12'))
    args=p.parse_args()
    if args.height<0:raise ValueError('Flight height must be nonnegative')
    source=args.input if args.input.is_absolute() else ROOT/args.input
    out=args.output_root if args.output_root.is_absolute() else ROOT/args.output_root
    out.mkdir(parents=True,exist_ok=True)
    xy,values,z=survey(source,args.channel)
    grid=prepare(xy,values)
    np.savez_compressed(out/'Thunderbird_Gzz_81x81.npz',**grid)
    map_plots(grid,xy,values,out)
    meta=dict(source=str(source),channel=args.channel,channel_name='Gzz (assumed)',
        unit='Eotvos (assumed)',n_unique_stations=len(xy),
        original_bounds_m=grid['bounds_m'].tolist(),
        original_width_m=float(np.ptp(xy[:,0])),original_height_m=float(np.ptp(xy[:,1])),
        observation_grid=[81,81],actual_receiver_spacing_m=[grid['actual_dx_m'],grid['actual_dy_m']],
        stretch_factors=[1.,1.],coverage_fraction=float(grid['coverage_mask'].mean()),
        unsupported_nodes=int(np.count_nonzero(~grid['coverage_mask'])),
        nearest_station_distance_max_inside_m=float(np.max(grid['nearest_station_distance_m'][grid['coverage_mask']])),
        original_z_range=[float(z.min()),float(z.max())],
        nominal_flight_height_m=args.height,
        height_source='Jorgensen and Zhdanov 2021, Minerals 11(4), 366, section 6.2: 100m nominal; not demonstrated constant above common datum',
        paper_line_spacing_m=250.,
        interpolation='linear within original convex hull; no extrapolation',
        physical_model_extent_m=[8000,8000,1000])
    (out/'preprocessing_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
    if args.experiment and not args.prepare_only:
        dest=out/args.experiment;dest.mkdir(parents=True,exist_ok=True)
        infer(grid,args.experiment,args.height,dest)
    else:print('Prepared observations only. No migration or inversion.')
if __name__=='__main__':main()
