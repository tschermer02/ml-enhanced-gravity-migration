"""Compare Thunderbird E08D and E10D mosaics with matched visualization and Gzz fits.

Run from repository root:
 python compare_thunderbird_mosaics.py \
   --e08 outputs/ml/field/Thunderbird_E08D \
   --e10 outputs/ml/field/Thunderbird_E10D_mosaic \
   --survey src/data/Thunderbird_gravity_data/Tbird_gravity.dat

Produces matched Plotly isosurfaces, cross-section PNG, gravity and residual maps,
and a JSON evaluation summary. The Gzz comparison is approximate because
legacy synthetic forward calculations assume an observation plane at z=0.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(root):
    root = Path(root)
    fields = {key: np.load(root / filename, mmap_mode='r' if key == 'rho' else None)
              for key, filename in {
                  'rho': 'density_3d_gcm3.npy',
                  'valid': 'coverage_mask.npy',
                  'x': 'x_cell_centers_m.npy',
                  'y': 'y_cell_centers_m.npy',
                  'z': 'z_cell_centers_m.npy',
              }.items()}
    if fields['rho'].shape != (len(fields['z']), len(fields['y']), len(fields['x'])):
        raise ValueError(f'{root}: inconsistent model and coordinate dimensions')
    if fields['valid'].shape != fields['rho'].shape[1:]:
        raise ValueError(f'{root}: invalid coverage mask')
    return fields


def assert_identical_grid(a, b):
    for key in ('x', 'y', 'z'):
        if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=0, atol=1e-5):
            raise ValueError(f'Mosaic coordinate grid mismatch for {key}; cannot compare without regridding')
    if a['rho'].shape != b['rho'].shape:
        raise ValueError('Mosaic density grids differ in shape')
    common = np.asarray(a['valid'], bool) & np.asarray(b['valid'], bool)
    if not np.any(common):
        raise ValueError('No shared valid coverage')
    return common


def plotly_matched(a, b, common, output, stride, threshold, n_surfaces):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Same physical positions, sign thresholds, color scales, and camera on both sides.
    xs = np.asarray(a['x'][::stride], dtype=np.float64)
    ys = np.asarray(a['y'][::stride], dtype=np.float64)
    zs = np.asarray(a['z'], dtype=np.float64)
    Z, Y, X = np.meshgrid(zs, (ys - a['y'][0])/1000., (xs - a['x'][0])/1000., indexing='ij')
    valid = common[::stride, ::stride]
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=('E08D · Per-window normalized', 'E10D · Fixed training scale'),
                        horizontal_spacing=0.03)
    colors = [(1, 'Reds', 'Positive'), (-1, 'Blues', 'Negative')]
    counts = {}
    for col, (name, d) in enumerate([('E08D', a), ('E10D', b)], start=1):
        rho = np.asarray(d['rho'][:, ::stride, ::stride], dtype=np.float32)
        counts[name] = {}
        for sign, palette, sign_label in colors:
            values = np.where(valid[None], np.maximum(sign * rho, 0.0), 0.0)
            maximum = float(values.max())
            counts[name][sign_label.lower()] = int(np.count_nonzero((sign * rho >= threshold) & valid[None]))
            if maximum <= threshold:
                continue
            fig.add_trace(go.Isosurface(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=values.ravel(),
                isomin=threshold, isomax=maximum, surface_count=n_surfaces,
                opacity=0.48, colorscale=palette, showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=f'{name} {sign_label}', showlegend=True,
                hovertemplate='X=%{x:.2f} km<br>Y=%{y:.2f} km<br>Depth=%{z:.0f} m<br>|Density|=%{value:.3f} g/cm³<extra></extra>',
            ), row=1, col=col)
    common_scene = dict(
        xaxis_title='X from survey west (km)',
        yaxis_title='Y from survey south (km)',
        zaxis_title='Depth (m)', zaxis=dict(autorange='reversed'),
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=.38),
        camera=dict(eye=dict(x=1.65, y=1.65, z=.85)),
    )
    fig.update_layout(scene=common_scene, scene2=common_scene,
                      title=f'Thunderbird mosaics · matched density threshold ±{threshold:g} g/cm³',
                      width=1560, height=800,
                      margin=dict(l=0, r=0, t=75, b=0))
    filename = output/'E08D_vs_E10D_matched_3d.html'
    fig.write_html(str(filename), include_plotlyjs=True)
    print('Wrote', filename)
    return counts


def plot_sections(a, b, common, output, vmax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    x = (np.asarray(a['x'])-a['x'][0])/1000.
    y = (np.asarray(a['y'])-a['y'][0])/1000.
    z = np.asarray(a['z'])
    j, i, k = len(y)//2, len(x)//2, min(8, len(z)-1)
    figures = [np.asarray(a['rho'], np.float32), np.asarray(b['rho'], np.float32)]
    fig, axes = plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
    sections = [('XY',k,'X (km)','Y (km)'),('XZ',j,'X (km)','Depth (m)'),('YZ',i,'Y (km)','Depth (m)')]
    for row, (name, vol) in enumerate(zip(['E08D','E10D'], figures)):
        for col,(plane,idx,xlabel,ylabel) in enumerate(sections):
            if plane=='XY':
                panel = np.where(common,vol[idx],np.nan)
                extent=(x[0],x[-1],y[0],y[-1]); origin='lower'
            elif plane=='XZ':
                panel = np.where(common[j][None],vol[:,j,:],np.nan)
                extent=(x[0],x[-1],z[-1],z[0]);origin='upper'
            else:
                panel = np.where(common[:,i][None],vol[:,:,i],np.nan)
                extent=(y[0],y[-1],z[-1],z[0]);origin='upper'
            im=axes[row,col].imshow(panel,cmap='RdBu_r',norm=norm,extent=extent,origin=origin,aspect='auto')
            axes[row,col].set(xlabel=xlabel,ylabel=ylabel,title=f'{name}: {plane}'+(f' at {z[k]:.0f} m' if plane=='XY' else ''))
    fig.colorbar(im,ax=axes[:,:].ravel().tolist(),label='Predicted density (g/cm³)',shrink=.65)
    dst=output/'E08D_vs_E10D_sections.png';fig.savefig(dst,dpi=180);plt.close(fig)
    print('Wrote',dst)


def read_manifest(root):
    with (Path(root)/'tile_manifest.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def gzz_validation(a,b,common,root_a,root_b,survey,output,number,channel,max_station_distance):
    """Matched independent spot-check of forward Gzz for selected tiled predictions.

    Reads the original observations to create an identical Gzz grid for each tile,
    forward-models mosaic density patch at matching locations, and reports residuals.
    This is the E08-style synthetic operator, NOT airborne-corrected physics.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import cKDTree
    from src.physics.gravity_forward import OBS_X, OBS_Y, calculate_gravity

    m_a = read_manifest(root_a)
    m_b = read_manifest(root_b)
    positions_b = {(round(float(r['x_origin_m']),5),round(float(r['y_origin_m']),5)) for r in m_b}
    matched = [r for r in m_a if (round(float(r['x_origin_m']),5),round(float(r['y_origin_m']),5)) in positions_b]
    if len(matched)==0:
        raise ValueError('No matching mosaic tile origins for residual calculation')
    # Evenly spaced tile sample, sorted in manifest order. Limit to control runtime.
    ids=np.unique(np.linspace(0,len(matched)-1,min(number,len(matched)),dtype=int))
    selected=[matched[i] for i in ids]

    data=np.loadtxt(survey)
    if data.ndim!=2 or data.shape[1]!=5:
        raise ValueError('Survey data must have 5 columns')
    points=data[data[:,3]==channel]
    if len(points)<10:
        raise ValueError('No survey data for requested channel')
    interp=LinearNDInterpolator(points[:,:2],points[:,4],fill_value=np.nan)
    near=NearestNDInterpolator(points[:,:2],points[:,4])
    tree=cKDTree(points[:,:2])
    x_start=float(a['x'][0]-5)
    y_start=float(a['y'][0]-5)
    rows=[]
    data_by_tile=[]
    for r in selected:
        x0=float(r['x_origin_m']);y0=float(r['y_origin_m'])
        ix=int(round((x0-x_start)/10));iy=int(round((y0-y_start)/10))
        if ix<0 or iy<0 or ix+64>a['rho'].shape[2] or iy+64>a['rho'].shape[1]:
            print('Skipping partial edge tile for comparable forward-model evaluation:',r['tile'])
            continue
        if not common[iy:iy+64,ix:ix+64].all():
            print('Skipping incompletely covered tile:',r['tile'])
            continue
        gx,gy=np.meshgrid(x0+OBS_X,y0+OBS_Y)
        pts=np.column_stack([gx.ravel(),gy.ravel()])
        station_dist,_=tree.query(pts)
        obs=np.asarray(interp(pts),dtype=np.float64)
        bad=~np.isfinite(obs)
        if bad.any():obs[bad]=near(pts[bad])
        obs=obs.reshape(81,81)
        reliable=(station_dist.reshape(81,81)<=max_station_distance)
        if reliable.sum()<100:
            print('Skipping poorly sampled tile:',r['tile'])
            continue
        patches=[np.asarray(vol['rho'][:,iy:iy+64,ix:ix+64],np.float32) for vol in (a,b)]
        calc=[calculate_gravity(p,component='gzz') for p in patches]
        res=[obs-c for c in calc]
        rmse=[float(np.sqrt(np.mean((rr[reliable])**2))) for rr in res]
        mae=[float(np.mean(np.abs(rr[reliable]))) for rr in res]
        rows.append({'tile':r['tile'],'x_origin_m':x0,'y_origin_m':y0,
                     'evaluated_nodes':int(reliable.sum()),'max_nearest_station_distance_m':float(station_dist.max()),
                     'e08_rmse_E':rmse[0],'e10_rmse_E':rmse[1],
                     'e08_mae_E':mae[0],'e10_mae_E':mae[1]})
        data_by_tile.append((r['tile'],obs,calc,res,reliable))
        print(f"Gzz {r['tile']}: E08 RMSE={rmse[0]:.4g} E; E10 RMSE={rmse[1]:.4g} E",flush=True)

    if not rows:
        print('No sufficiently covered tiles for Gzz residual check; skipping.')
        return {'status':'no eligible tiles'}
    with (output/'matched_tile_Gzz_RMSE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    avg={key:float(np.mean([r[key] for r in rows])) for key in ('e08_rmse_E','e10_rmse_E','e08_mae_E','e10_mae_E')}

    # Make a 2D residual comparison for the first selected tile at matched color scales.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    tile,obs,calc,res,reliable=data_by_tile[0]
    values=[obs,calc[0],calc[1],res[0],res[1]]
    common_lim=max(float(np.percentile(np.abs(v[reliable]),99)) for v in values[:3])
    resid_lim=max(float(np.percentile(np.abs(v[reliable]),99)) for v in values[3:])
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    plot_values=[obs,calc[0],calc[1],np.full_like(obs,np.nan),res[0],res[1]]
    labels=['Observed Gzz','E08D forward Gzz','E10D forward Gzz','',
            'E08D residual','E10D residual']
    for ax,val,label in zip(axes.flat,plot_values,labels):
        if not label:
            ax.axis('off');continue
        v=np.where(reliable,val,np.nan)
        bound=resid_lim if 'residual' in label else common_lim
        bound=max(bound,1e-9)
        im=ax.imshow(v,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),origin='lower',
                     extent=(-85,715,-85,715))
        ax.set(title=label,xlabel='Tile-relative X (m)',ylabel='Tile-relative Y (m)')
        fig.colorbar(im,ax=ax,shrink=.8,label='Gzz (Eötvös)')
    fig.suptitle(f'Gzz comparison at {tile} (nodes within {max_station_distance:g} m of stations)')
    dst=output/'Gzz_observed_forward_residual_comparison.png'
    fig.savefig(dst,dpi=180);plt.close(fig)
    print('Wrote',dst)
    return {'evaluated_tiles':len(rows),'mean_tile_metrics':avg,
            'note':'Unweighted mean of tile RMSE, computed only at observation grid nodes within max station distance; mosaic forward model ignores density outside the 640m window and altitude mismatch.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e08',type=Path,default=Path('outputs/ml/field/Thunderbird_E08D'))
    p.add_argument('--e10',type=Path,default=Path('outputs/ml/field/Thunderbird_E10D_mosaic'))
    p.add_argument('--survey',type=Path,default=Path('src/data/Thunderbird_gravity_data/Tbird_gravity.dat'))
    p.add_argument('--output',type=Path,default=Path('outputs/ml/field/Thunderbird_E08D_vs_E10D'))
    p.add_argument('--stride',type=int,default=6,help='Plotly XY visualization decimation only')
    p.add_argument('--threshold',type=float,default=.1,help='Shared absolute density iso-threshold, g/cm3')
    p.add_argument('--surfaces',type=int,default=4)
    p.add_argument('--gzz-tiles',type=int,default=9,help='Number of evenly-spaced matched mosaic tiles for Gzz residual check')
    p.add_argument('--max-station-distance',type=float,default=180.,help='Only evaluate nodes within this distance of an observed station')
    p.add_argument('--skip-gzz',action='store_true')
    args=p.parse_args()
    if args.stride<1 or args.threshold<=0 or args.surfaces<1 or args.gzz_tiles<1:
        p.error('Invalid numeric option')
    a=load(args.e08);b=load(args.e10)
    common=assert_identical_grid(a,b)
    print('Matched mosaic dimensions:',a['rho'].shape)
    print('Coverage in common:',float(common.mean()))
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    counts=plotly_matched(a,b,common,output,args.stride,args.threshold,args.surfaces)
    plot_sections(a,b,common,output,vmax=1.)
    summary={'e08_path':str(args.e08),'e10_path':str(args.e10),
             'shape':list(a['rho'].shape),'shared_coverage_fraction':float(common.mean()),
             'threshold_gcm3':args.threshold,'plotly_sampled_voxel_counts_above_threshold':counts,
             'visualization_stride':args.stride}
    valid3d=np.broadcast_to(common,a['rho'].shape)
    for name,dat in [('E08D',a),('E10D',b)]:
        rho=dat['rho']
        summary[name]={'mean_density_gcm3':float(np.mean(rho[:,common])),
                       'mean_absolute_density_gcm3':float(np.mean(np.abs(rho[:,common]))),
                       'positive_fraction_above_threshold':float(np.mean(rho[:,common]>args.threshold)),
                       'negative_fraction_below_threshold':float(np.mean(rho[:,common]<-args.threshold))}
    if not args.skip_gzz:
        summary['gzz_checks']=gzz_validation(a,b,common,args.e08,args.e10,args.survey,output,args.gzz_tiles,7,args.max_station_distance)
    (output/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
    print('Saved comparison summary:',output/'comparison_summary.json')
    print('NOTE: Synthetic forward model uses z_obs=0, and does not account for airborne altitude or out-of-tile density.')

if __name__=='__main__':main()
