"""Full-survey Thunderbird E08D Gzz inference; run in repo root.

Processes every Gzz observation as part of a continuous interpolation field,
then predicts overlapping 640 x 640 x 240 m density blocks using pretrained E08D.
This is inference, NOT retraining and NOT a measured-data misfit inversion.

Example:
 python run_thunderbird_full_e08d.py --input Tbird_gravity.dat \
   --model outputs/ml/training/E08D/checkpoints/best_unet3d.keras \
   --output-root outputs/ml/field/Thunderbird_E08D --stride 480
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from src.physics.gravity_forward import OBS_X, OBS_Y, CELL_SIZE
from src.physics.gravity_migration import calculate_migration, normalize_migration


def tiles(minimum, maximum, stride):
    # Model spans 64 cells of 10 m each. Include the final partial tile.
    n = max(1, math.ceil((maximum - minimum - 640) / stride) + 1)
    return [minimum + i * stride for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('outputs/ml/field/Thunderbird_E08D'))
    p.add_argument('--channel', type=int, default=7, help='Gzz channel; verify input metadata')
    p.add_argument('--stride', type=int, default=480, help='Tile spacing in metres, multiple of 10, <= 640')
    p.add_argument('--max-distance', type=float, default=180.,
                   help='Flag grid nodes more than this distance (m) from measured Gzz stations')
    p.add_argument('--save-tiles', action='store_true', help='Save each tile prediction; increases disk use')
    p.add_argument('--limit', type=int, default=None, help='Debug: first N tiles, no complete mosaic')
    args = p.parse_args()
    if args.stride <= 0 or args.stride > 640 or args.stride % 10:
        p.error('--stride must be a 10 m multiple between 10 and 640')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be >=1')
    if not args.input.exists() or not args.model.exists():
        p.error('Missing --input or --model file')

    data = np.loadtxt(args.input)
    if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
        raise ValueError('Expected five finite columns: x y z component value')
    field = data[data[:,3] == args.channel]
    if not len(field):
        raise ValueError(f'No component {args.channel} observations')
    if np.ptp(field[:,2]) > 1e-7:
        raise ValueError('Varying survey z is not supported by this flat-plane E08 forward model')
    xy, g = field[:,:2], field[:,4]
    if len(np.unique(xy,axis=0)) != len(xy):
        raise ValueError('Duplicate XY locations: average or resolve them before interpolating')
    # Tile origins and assembled model coordinates are on a 10 m spatial grid.
    x_start = 10 * math.floor(xy[:,0].min()/10)
    y_start = 10 * math.floor(xy[:,1].min()/10)
    x_end = 10 * math.ceil(xy[:,0].max()/10)
    y_end = 10 * math.ceil(xy[:,1].max()/10)
    nx = int((x_end-x_start)/10)
    ny = int((y_end-y_start)/10)
    x_origins, y_origins = tiles(x_start,x_end,args.stride), tiles(y_start,y_end,args.stride)
    total = len(x_origins)*len(y_origins)
    print(f'Gzz stations: {len(field)} | model extent: {nx} x {ny} x 24 voxels')
    print(f'Tiles: {len(x_origins)} columns x {len(y_origins)} rows = {total}; stride={args.stride} m')
    print('Building linear interpolator from ALL Gzz observations...', flush=True)
    linear = LinearNDInterpolator(xy, g, fill_value=np.nan)
    nearest = NearestNDInterpolator(xy, g)
    tree = cKDTree(xy)
    # Loading without compilation avoids custom losses on prediction-only runs.
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model, compile=False)
    if tuple(model.input_shape[1:]) != (24,64,64,1):
        raise ValueError(f'Expected E08D input (24,64,64,1), got {model.input_shape}')
    if tuple(model.output_shape[1:]) != (24,64,64,1):
        raise ValueError(f'Expected E08D output (24,64,64,1), got {model.output_shape}')

    out = args.output_root
    out.mkdir(parents=True,exist_ok=True)
    weights = np.zeros((ny,nx), dtype=np.float32)
    accumulator = np.zeros((24,ny,nx),dtype=np.float32)
    # Raised-cosine taper for overlapping predictions; positive on tile boundaries.
    axis = np.maximum(np.sin(np.pi*(np.arange(64)+0.5)/64)**2, 0.03).astype(np.float32)
    taper = axis[:,None]*axis[None,:]
    manifest=[]
    encountered_far=0
    observed_points=0
    k=0
    for iy,y0 in enumerate(y_origins):
      for ix,x0 in enumerate(x_origins):
        k += 1
        # Forward physics expects observations x=-85..715 relative to model x0.
        gx,gy=np.meshgrid(x0+OBS_X,y0+OBS_Y,indexing='xy')
        pts=np.column_stack([gx.ravel(),gy.ravel()])
        vals=linear(pts)
        absent=~np.isfinite(vals)
        if np.any(absent):
            vals[absent]=nearest(pts[absent])
        d, _ = tree.query(pts)
        encountered_far += int(np.count_nonzero(d > args.max_distance))
        observed_points += len(d)
        gravity=vals.reshape(81,81).astype(np.float32)
        migration=calculate_migration(gravity,component='gzz',restrict=False)
        migration_input,scale=normalize_migration(migration)
        pred=model(migration_input[None,...,None],training=False)
        pred=np.asarray(pred)[0,...,0].astype(np.float32)
        if pred.shape != (24,64,64) or not np.isfinite(pred).all():
            raise ValueError(f'Invalid model output for tile {k}')
        xo,yo=int((x0-x_start)/10),int((y0-y_start)/10)
        xs=slice(xo,min(nx,xo+64));ys=slice(yo,min(ny,yo+64))
        w=taper[:ys.stop-ys.start,:xs.stop-xs.start]
        weights[ys,xs]+=w
        accumulator[:,ys,xs]+=pred[:,:w.shape[0],:w.shape[1]]*w
        tile_name=f'tile_y{iy:03d}_x{ix:03d}'
        if args.save_tiles:
            folder=out/'tiles';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{tile_name}.npz',
                 gravity=gravity,migration_input=migration_input,
                 prediction=pred,migration_scale=scale,origin_xy=np.array([x0,y0]))
        manifest.append(dict(tile=tile_name,x_origin_m=x0,y_origin_m=y0,
                             max_station_distance_m=round(float(d.max()),2),
                             nearest_filled_points=int(np.sum(absent)),
                             gzz_min=float(gravity.min()),gzz_max=float(gravity.max()),
                             migration_scale=float(scale)))
        if k % 10 == 0 or k == total:
            print(f'Predicted {k}/{total} tiles',flush=True)
        if args.limit is not None and k >= args.limit:
            break
      if args.limit is not None and k >= args.limit:
          break

    with (out/'tile_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=manifest[0].keys());writer.writeheader();writer.writerows(manifest)
    covered=weights>0
    density=np.zeros_like(accumulator)
    np.divide(accumulator,weights[None,:,:],out=density,where=covered[None,:,:])
    np.save(out/'density_3d_gcm3.npy',density)
    np.save(out/'coverage_mask.npy',covered)
    np.save(out/'x_cell_centers_m.npy',x_start+5+10*np.arange(nx))
    np.save(out/'y_cell_centers_m.npy',y_start+5+10*np.arange(ny))
    np.save(out/'z_cell_centers_m.npy',5+10*np.arange(24))
    info=dict(input=str(args.input),checkpoint=str(args.model),channel=args.channel,
        assumed_component='Gzz',assumed_unit='Eotvos (UNVERIFIED)',
        station_count=len(field),data_columns='x,y,z,channel,value',
        observed_z=float(field[0,2]),model_shape=list(density.shape),
        model_cell_size_m=10,model_origin_xy=[x_start,y_start],
        stride_m=args.stride,tiles_completed=k,tiles_expected=total,
        complete=(k==total),max_station_distance_limit_m=args.max_distance,
        fraction_grid_samples_far_from_station=encountered_far/observed_points,
        warning='Pretrained synthetic model and coarse irregular field survey; predictions not validated density inversion.')
    (out/'run_metadata.json').write_text(json.dumps(info,indent=2))
    print(f'Saved {out/"density_3d_gcm3.npy"}; shape={density.shape}')
    print(f'Coverage: {covered.mean():.2%}; fraction station distance > {args.max_distance:.0f}m: {encountered_far/observed_points:.2%}')
    if k != total: print('WARNING: limited debug run, mosaic incomplete')
    if not covered.all() and k==total: print('WARNING: some model pixels not covered; inspect coverage_mask.npy')


if __name__=='__main__':
    main()
