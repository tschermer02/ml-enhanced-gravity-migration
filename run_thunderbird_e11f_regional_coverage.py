"""Coverage-aware Thunderbird regional resampling for E11F.

Uses only observed Gzz channel values and linear interpolation INSIDE the surveyed
convex hull. A centered inset rectangle is chosen if necessary to avoid creating
receiver values by extrapolation. Gridded interpolation is not an observation.

Run in repository root:
  python run_thunderbird_e11f_regional_coverage.py --input-unit eotvos

Requires repository src modules and trained E11F checkpoint. Does not retrain.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree, Delaunay

from src.data.run_thunderbird_e11 import load_field, predict
from src.physics.gravity_forward import (
    E11_DX_M, E11_DY_M, E11_DZ_M, OBS_SHAPE, SHAPE,
    observation_coordinates, calculate_gzz,
)
from src.physics.gravity_migration import calculate_migration, normalize_migration
from src.visualization.plotly_3d import plot_e11_regional_3d


def plot_coverage(path, original_xy, survey_min, survey_max, raw_covered,
                  final_bounds, final_xy, final_covered, nearest_dist, max_distance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    # Original survey in source coordinates (decimated drawing only).
    skip = max(1, len(original_xy) // 25000)
    axs[0].scatter(original_xy[::skip, 0]/1000, original_xy[::skip, 1]/1000,
                   s=0.4, alpha=0.35, color='#264f83', rasterized=True)
    (xmin, ymin), (xmax, ymax) = final_bounds
    axs[0].add_patch(Rectangle((xmin/1000, ymin/1000),
                               (xmax-xmin)/1000, (ymax-ymin)/1000,
                               fill=False, lw=2, color='#c04d3e', label='Retained survey rectangle'))
    axs[0].set(title='Original measured stations and retained extent',
               xlabel='Survey X (km)', ylabel='Survey Y (km)')
    axs[0].set_aspect('equal', adjustable='box'); axs[0].legend(loc='best', fontsize=8)
    axs[1].imshow(raw_covered, origin='lower', interpolation='nearest', cmap='RdYlGn',vmin=0,vmax=1)
    axs[1].set(title='Coverage if full bounding box were used',xlabel='E11 receiver column',ylabel='E11 receiver row')
    axs[2].imshow(final_covered, origin='lower', interpolation='nearest', cmap='RdYlGn',vmin=0,vmax=1)
    im=axs[2].contour(nearest_dist, levels=[max_distance], colors=['black'], linewidths=1.0)
    axs[2].set(title='Retained area: measured-data support (green)',xlabel='E11 receiver column',ylabel='E11 receiver row')
    fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=Path('src/data/Thunderbird_gravity_data/Tbird_gravity.dat'))
    p.add_argument('--input-unit',choices=['eotvos','si'],required=True)
    p.add_argument('--channel',type=int,default=7,help='Assumed Gzz channel; verify against original survey metadata')
    p.add_argument('--height',type=float,default=100.,help='Assumed E11 flat receiver height (m)')
    p.add_argument('--output-root',type=Path,default=Path('outputs/ml/field/Thunderbird_E11/E11F_regional_coverage'))
    p.add_argument('--checkpoint',type=Path,default=Path('outputs/ml/training/E11F/checkpoints/best_unet3d.keras'))
    p.add_argument('--distance-factor',type=float,default=4.,help='Maximum nearest-station distance / median original station spacing')
    p.add_argument('--min-retained-fraction',type=float,default=0.50,help='Minimum allowed retained fraction along each original XY axis')
    p.add_argument('--density-threshold',type=float,default=0.1)
    p.add_argument('--migration-percentile',type=float,default=95.)
    p.add_argument('--max-points',type=int,default=18000)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if not (args.input.exists() and args.checkpoint.exists()):
        raise FileNotFoundError('Missing Thunderbird data file or E11F checkpoint')
    if args.height<0 or args.distance_factor<=0 or not (0<args.min_retained_fraction<=1):
        p.error('Invalid height, distance-factor or retained fraction')
    out=args.output_root
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{out} exists and is nonempty; pass --overwrite')

    xy, values, source_z=load_field(args.input,args.channel,args.input_unit)
    survey_min=np.min(xy,axis=0);survey_max=np.max(xy,axis=0)
    span=survey_max-survey_min
    if np.any(span<=0):raise ValueError('Invalid survey span')
    center=(survey_max+survey_min)/2
    obs_x,obs_y=observation_coordinates(E11_DX_M,E11_DY_M)
    rx,ry=np.meshgrid(obs_x,obs_y)
    rxy=np.column_stack((rx.ravel(),ry.ravel()))
    rmin=np.array([obs_x.min(),obs_y.min()]);rspan=np.array([np.ptp(obs_x),np.ptp(obs_y)])
    relative=(rxy-rmin)/rspan # 0..1; used to sample original survey, no station values changed
    station_tree=cKDTree(xy)
    nearest_neighbor=station_tree.query(xy,k=2)[0][:,1]
    source_spacing=float(np.median(nearest_neighbor[nearest_neighbor>0]))
    max_distance=args.distance_factor*source_spacing
    tri=Delaunay(xy)

    def sample_coverage(fraction):
        bounds_min=center-span*fraction/2
        bounds_max=center+span*fraction/2
        points=bounds_min+relative*(bounds_max-bounds_min)
        hull_ok=tri.find_simplex(points,tol=1e-10)>=0
        distances=station_tree.query(points)[0]
        covered=hull_ok & (distances<=max_distance)
        return bounds_min,bounds_max,points,covered,distances,hull_ok

    _,_,_,full_covered,_,_=sample_coverage(1.)
    # Narrow only the geographic footprint; NEVER fill unknown receiver readings.
    # All receivers must lie inside survey convex hull and sufficiently close to data.
    selected=None
    for fraction in np.linspace(1.,args.min_retained_fraction,101):
        candidate=sample_coverage(float(fraction))
        if candidate[3].all():
            selected=(float(fraction),candidate)
            break
    if selected is None:
        out.mkdir(parents=True,exist_ok=True)
        plot_coverage(out/'interpolation_coverage_map.png',xy,survey_min,survey_max,
                      full_covered.reshape(OBS_SHAPE),
                      np.vstack((survey_min,survey_max)),
                      rxy,np.zeros(OBS_SHAPE,bool),np.zeros(OBS_SHAPE),max_distance)
        raise RuntimeError('No fully covered rectangular receiver footprint found. '
                           'Coverage map was saved. No migration or inference was run. '
                           'Try increasing --distance-factor only if survey spacing justifies it; '
                           'otherwise examine survey coverage and accept a smaller region deliberately.')
    fraction,(bmin,bmax,source_targets,covered,distances,hull_ok)=selected
    linear=LinearNDInterpolator(xy,values,fill_value=np.nan)
    sampled=np.asarray(linear(source_targets),dtype=np.float64)
    if not np.isfinite(sampled).all():
        raise RuntimeError('Linear interpolation missing at retained receiver sites; refusing nearest fill.')
    gravity=sampled.reshape(OBS_SHAPE).astype('float32')
    scale_xy=rspan/(bmax-bmin)
    retained_original_fraction=float(np.prod((bmax-bmin)/span))
    out.mkdir(parents=True,exist_ok=True)
    plot_coverage(out/'interpolation_coverage_map.png',xy,survey_min,survey_max,
                  full_covered.reshape(OBS_SHAPE),np.vstack((bmin,bmax)),
                  source_targets,covered.reshape(OBS_SHAPE),
                  distances.reshape(OBS_SHAPE),max_distance)

    migration=calculate_migration(gravity,component='gzz',restrict=False,
       dx_m=E11_DX_M,dy_m=E11_DY_M,dz_m=E11_DZ_M,observation_height_m=args.height)
    migration_input,migration_scale=normalize_migration(migration)
    density=predict(args.checkpoint,migration_input)
    modeled=calculate_gzz(density,dx_m=E11_DX_M,dy_m=E11_DY_M,
        dz_m=E11_DZ_M,observation_height_m=args.height).astype('float32')
    residual=gravity-modeled
    rmse=float(np.sqrt(np.mean(residual**2)))
    x=(np.arange(64)+.5)*E11_DX_M;y=(np.arange(64)+.5)*E11_DY_M
    z=(np.arange(24)+.5)*E11_DZ_M
    np.savez_compressed(out/'Thunderbird_E11F_regional.npz',
       gravity=gravity,migration=migration.astype('float32'),
       migration_input=migration_input,prediction=density,
       modeled_gravity=modeled,gravity_residual=residual,
       migration_scale=np.float64(migration_scale),
       receiver_supported_mask=covered.reshape(OBS_SHAPE),
       original_survey_xy_min_m=survey_min,original_survey_xy_max_m=survey_max,
       retained_survey_xy_min_m=bmin,retained_survey_xy_max_m=bmax,
       survey_to_model_scale_xy=scale_xy,
       original_receiver_source_xy_m=source_targets.reshape((*OBS_SHAPE,2)),
       original_station_nearest_distance_m=distances.reshape(OBS_SHAPE),
       max_station_distance_m=np.float64(max_distance),
       observation_height_m=np.float64(args.height),
       x_cell_centers_m=x,y_cell_centers_m=y,z_cell_centers_m=z)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig,axs=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    clim=max(float(np.percentile(np.abs(np.r_[gravity.ravel(),modeled.ravel()]),99)),1e-8)
    rlim=max(float(np.percentile(np.abs(residual),99)),1e-8)
    for ax,arr,title,lim in zip(axs,(gravity,modeled,residual),
        ('Interpolated measured Gzz','E11F forward Gzz','Observed − forward Gzz'),(clim,clim,rlim)):
        im=ax.imshow(arr,origin='lower',extent=(obs_x.min()/1000,obs_x.max()/1000,obs_y.min()/1000,obs_y.max()/1000),
                     cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='equal')
        ax.set(xlabel='E11 receiver X (km)',ylabel='E11 receiver Y (km)',title=title)
        fig.colorbar(im,ax=ax,label='Eötvös (E)',shrink=.8)
    fig.savefig(out/'regional_observed_forward_residual_Gzz.png',dpi=220);plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
    for row,(volume,name) in enumerate(((migration_input,'Normalized Gzz migration'),(density,'Predicted density (g/cm³)'))):
        lim=max(float(np.percentile(np.abs(volume),99)),1e-8)
        for col,(image,extent,label) in enumerate(((volume[7],(0,8,0,8),'XY slice'),
              (volume[:,32,:],(0,8,1,0),'XZ slice'),(volume[:,:,32],(0,8,1,0),'YZ slice'))):
            ax=axs[row,col]
            im=ax.imshow(image,origin='lower' if col==0 else 'upper',extent=extent,
                 cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
            ax.set(title=f'{name}: {label}',xlabel='X (km)' if col<2 else 'Y (km)',
                   ylabel='Y (km)' if col==0 else 'Depth (km)')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(out/'regional_migration_vs_density_sections.png',dpi=220);plt.close(fig)
    plot_e11_regional_3d(migration_input,density,
        out/'regional_migration_vs_density_3D.html',
        x_cell_centers_m=x,y_cell_centers_m=y,z_cell_centers_m=z,
        threshold=args.density_threshold,migration_percentile=args.migration_percentile,
        max_points=args.max_points,make_gif=False)
    meta={
      'experiment':'E11F','source':str(args.input),'channel':args.channel,
      'input_unit':args.input_unit,'source_z_coordinate':source_z,
      'assumed_receiver_height_m':args.height,
      'original_survey_span_xy_m':span.tolist(),
      'retained_survey_span_xy_m':(bmax-bmin).tolist(),
      'retained_fraction_of_each_original_axis':fraction,
      'retained_fraction_of_original_bounding_box_area':retained_original_fraction,
      'trained_receiver_span_xy_m':rspan.tolist(),
      'survey_to_e11_coordinate_scale_xy':scale_xy.tolist(),
      'survey_original_bounds_xy_m':[survey_min.tolist(),survey_max.tolist()],
      'survey_retained_bounds_xy_m':[bmin.tolist(),bmax.tolist()],
      'original_station_spacing_median_m':source_spacing,
      'maximum_allowed_distance_to_station_m':max_distance,
      'full_bounding_box_supported_fraction':float(full_covered.mean()),
      'retained_receiver_coverage_fraction':float(covered.mean()),
      'interpolation':'Linear interpolation strictly inside survey convex hull and station-distance coverage; never nearest-filled',
      'no_extrapolation':True,'synthetic_observations_added':False,
      'covered_receiver_gzz_rmse_e':rmse,
      'migration_normalization_scale':float(migration_scale),
      'model_extent_m':[8000,8000,1000],
      'interpretation':'Coordinates are rescaled. Linear interpolation does not create independent measurements. Model-scale depths/densities are not validated geologic properties.'}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('Original survey span (m):',span,'retained span (m):',bmax-bmin)
    print('Fraction of bounding rectangle retained:',f'{retained_original_fraction:.1%}')
    print('Survey -> E11 scale factors:',scale_xy)
    print('Full bounding-grid coverage:',f'{full_covered.mean():.1%}')
    print('Final receiver coverage:',f'{covered.mean():.1%}')
    print('Nearest-station median/max (m):',np.median(distances),np.max(distances))
    print('Measured-source Gzz resampled range (E):',np.min(gravity),np.max(gravity))
    print('Forward Gzz RMSE (E):',rmse)
    print('Saved:',out)

if __name__=='__main__':main()
