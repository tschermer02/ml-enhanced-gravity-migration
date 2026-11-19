"""Rescale Thunderbird Gzz regional survey onto E11F's trained receiver footprint.

Run from repository root:
  python run_thunderbird_e11f_regional.py --input-unit eotvos

Reuses the E11 forward/migration operators and trained E11F network without
changing them. This is a coordinate-rescaled, qualitative field inference;
rescaling does NOT increase measured resolution or preserve original geology.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from src.data.run_thunderbird_e11 import load_field, predict
from src.physics.gravity_forward import (
    E11_DX_M, E11_DY_M, E11_DZ_M, OBS_SHAPE, SHAPE,
    observation_coordinates, calculate_gzz,
)
from src.physics.gravity_migration import calculate_migration, normalize_migration
from src.visualization.plotly_3d import plot_e11_regional_3d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('src/data/Thunderbird_gravity_data/Tbird_gravity.dat'))
    p.add_argument('--input-unit', choices=('eotvos', 'si'), required=True,
                   help='Units of measured Gzz (must be checked against source data)')
    p.add_argument('--channel', type=int, default=7)
    p.add_argument('--height', type=float, default=100.,
                   help='Assumed flat observation height in E11 model coordinates (m)')
    p.add_argument('--output-root', type=Path,
                   default=Path('outputs/ml/field/Thunderbird_E11/E11F_regional_rescaled'))
    p.add_argument('--checkpoint', type=Path,
                   default=Path('outputs/ml/training/E11F/checkpoints/best_unet3d.keras'))
    p.add_argument('--density-threshold', type=float, default=0.1)
    p.add_argument('--migration-percentile', type=float, default=97.)
    p.add_argument('--max-points', type=int, default=18000)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.height < 0 or not np.isfinite(args.height): p.error('--height must be nonnegative')
    if not 0 < args.migration_percentile < 100: p.error('--migration-percentile must be in (0,100)')
    if args.density_threshold <= 0: p.error('--density-threshold must be positive')
    if not args.checkpoint.exists(): raise FileNotFoundError(args.checkpoint)
    if not args.input.exists(): raise FileNotFoundError(args.input)

    out=args.output_root
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{out} exists and is nonempty; use --overwrite')

    xy, values, source_z = load_field(args.input, args.channel, args.input_unit)
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    if not (xmax > xmin and ymax > ymin): raise ValueError('Survey X/Y extent must be nonzero')

    # Affine map of survey bounding rectangle onto the complete 81x81 receiver
    # footprint, NOT just the 8x8 km density model domain.
    ox, oy = observation_coordinates(E11_DX_M, E11_DY_M)
    receiver_min = np.array([ox.min(), oy.min()], dtype=float)
    receiver_max = np.array([ox.max(), oy.max()], dtype=float)
    survey_min = np.array([xmin, ymin], dtype=float)
    survey_max = np.array([xmax, ymax], dtype=float)
    scale = (receiver_max - receiver_min) / (survey_max - survey_min)
    scaled_xy = (xy - survey_min) * scale + receiver_min
    gx, gy = np.meshgrid(ox, oy)
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    linear = LinearNDInterpolator(scaled_xy, values, fill_value=np.nan)
    sampled = np.asarray(linear(targets), dtype=float)
    outside = ~np.isfinite(sampled)
    distances = cKDTree(scaled_xy).query(targets)[0]
    if np.any(outside):
        nearest = NearestNDInterpolator(scaled_xy, values)
        sampled[outside] = nearest(targets[outside])
    gravity = sampled.reshape(OBS_SHAPE).astype(np.float32)
    if not np.isfinite(gravity).all(): raise ValueError('Invalid resampled gravity')

    print(f'Survey extent: X={xmax-xmin:.1f} m, Y={ymax-ymin:.1f} m',flush=True)
    print(f'E11 receiver extent: X={ox.ptp() if hasattr(ox,"ptp") else np.ptp(ox):.1f} m, Y={np.ptp(oy):.1f} m', flush=True)
    print(f'Coordinate scale: X={scale[0]:.6g}, Y={scale[1]:.6g}',flush=True)
    print(f'Receiver sites outside survey convex hull: {outside.mean():.1%} (nearest-filled)',flush=True)
    print(f'Nearest-station distance in rescaled coordinates: median={np.median(distances):.1f} m, max={distances.max():.1f} m',flush=True)
    print(f'Gzz observed range: {gravity.min():.5g} to {gravity.max():.5g} E',flush=True)

    migration = calculate_migration(
        gravity,component='gzz',restrict=False,
        dx_m=E11_DX_M,dy_m=E11_DY_M,dz_m=E11_DZ_M,
        observation_height_m=args.height)
    migration_input,mig_scale=normalize_migration(migration)
    if migration_input.shape!=SHAPE: raise ValueError('Migration shape is not E11')
    density=predict(args.checkpoint,migration_input)
    modeled=calculate_gzz(density,dx_m=E11_DX_M,dy_m=E11_DY_M,
                          dz_m=E11_DZ_M,observation_height_m=args.height).astype(np.float32)
    residual=gravity-modeled
    valid=~outside.reshape(OBS_SHAPE)
    rmse=float(np.sqrt(np.mean(residual[valid]**2))) if np.any(valid) else None
    out.mkdir(parents=True,exist_ok=True)

    x=(np.arange(64)+.5)*E11_DX_M
    y=(np.arange(64)+.5)*E11_DY_M
    z=(np.arange(24)+.5)*E11_DZ_M
    np.savez_compressed(out/'Thunderbird_E11F_regional.npz',
        gravity=gravity,migration=np.asarray(migration,dtype=np.float32),
        migration_input=migration_input,prediction=density,
        modeled_gravity=modeled,gravity_residual=residual,
        migration_scale=np.float64(mig_scale),
        receiver_outside_hull_mask=outside.reshape(OBS_SHAPE),
        receiver_nearest_station_distance_m=distances.reshape(OBS_SHAPE),
        x_cell_centers_m=x,y_cell_centers_m=y,z_cell_centers_m=z,
        original_survey_xy_min_m=survey_min,original_survey_xy_max_m=survey_max,
        survey_to_model_scale_xy=scale,
        observation_height_m=np.float64(args.height))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    # Three directly comparable receiver-plane panels plus residual.
    fig, axs=plt.subplots(1,3,figsize=(16,5.1),constrained_layout=True)
    color_limit=max(float(np.percentile(np.abs(np.r_[gravity.ravel(),modeled.ravel()]),99)),1e-8)
    rlimit=max(float(np.percentile(np.abs(residual),99)),1e-8)
    for ax, arr, title, limit in zip(
        axs,(gravity,modeled,residual),
        ('Observed Gzz (rescaled)','E11F forward Gzz','Observed − forward Gzz'),
        (color_limit,color_limit,rlimit)):
        im=ax.imshow(arr,origin='lower',extent=(ox.min()/1000,ox.max()/1000,oy.min()/1000,oy.max()/1000),
                     cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit),aspect='equal')
        ax.set(xlabel='E11 receiver X (km)',ylabel='E11 receiver Y (km)',title=title)
        fig.colorbar(im,ax=ax,label='Eötvös (E)',shrink=.8)
    fig.suptitle('Thunderbird regional → E11F: Gzz response (coordinate-rescaled)')
    fig.savefig(out/'regional_observed_forward_residual_Gzz.png',dpi=220)
    plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
    zidx=7; yidx=32; xidx=32
    for row, (volume,title) in enumerate(((migration_input,'Normalized Gzz migration'),(density,'E11F density (g/cm³)'))):
        lim=max(float(np.percentile(np.abs(volume),99)),1e-8)
        views=(volume[zidx,:,:],volume[:,yidx,:],volume[:,:,xidx])
        extents=((0,8,0,8),(0,8,1,0),(0,8,1,0))
        names=(f'XY at {z[zidx]:.0f} m','XZ through center','YZ through center')
        for col,(v,extent,name) in enumerate(zip(views,extents,names)):
            ax=axs[row,col]
            im=ax.imshow(v,origin='lower' if col==0 else 'upper',extent=extent,
                         cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),aspect='auto')
            ax.set(xlabel='X (km)' if col<2 else 'Y (km)',
                   ylabel='Y (km)' if col==0 else 'Depth (km)',title=f'{title}\n{name}')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(out/'regional_migration_vs_density_sections.png',dpi=220)
    plt.close(fig)

    html=out/'regional_migration_vs_density_3D.html'
    plot_e11_regional_3d(migration_input,density,html,
         x_cell_centers_m=x,y_cell_centers_m=y,z_cell_centers_m=z,
         threshold=args.density_threshold,
         migration_percentile=args.migration_percentile,
         max_points=args.max_points,make_gif=False)
    meta={
        'experiment':'E11F','source':str(args.input),'source_z_coordinate':source_z,
        'model_checkpoint':str(args.checkpoint),'Gzz_channel':args.channel,
        'input_unit':args.input_unit,'assumed_height_m':args.height,
        'survey_bounds_xy_m':[[float(xmin),float(ymin)],[float(xmax),float(ymax)]],
        'survey_span_xy_m':[float(xmax-xmin),float(ymax-ymin)],
        'rescaled_receiver_span_xy_m':[float(np.ptp(ox)),float(np.ptp(oy))],
        'survey_to_e11_coordinate_scale_xy':scale.tolist(),
        'resampling':'affine map survey bounding box onto 81x81 E11 receiver footprint; linear scattered interpolation, nearest exterior fill',
        'outside_survey_hull_fraction':float(outside.mean()),
        'nearest_station_distance_median_in_model_coords':float(np.median(distances)),
        'nearest_station_distance_max_in_model_coords':float(np.max(distances)),
        'receiver_covered_Gzz_RMSE_E':rmse,
        'migration_normalization_scale':float(mig_scale),
        'model_extent_m':[8000.,8000.,1000.],
        'limitation':'Coordinate rescaling changes the physical relationship between survey wavelengths and subsurface depths. Figures are illustrative inference, not validated physical field density. Nearest-filled receivers are extrapolated, not observed.',
    }
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print('Saved outputs in',out,flush=True)
    print('Covered-receiver Gzz RMSE (rescaled):',rmse,'E',flush=True)
    print('Open interactive Plotly:',html,flush=True)

if __name__=='__main__':
    main()
