"""Visualize full Thunderbird E08D migration vs predicted density.

Run from repository root:
 python -m src.visualization.visualize_thunderbird_e08d \
   --output-root outputs/ml/field/Thunderbird_E08D

Creates cross_section_comparison.png and comparison_3d.html. Does not rerun the network.
Migration is reconstituted from Gzz stations on the same tile origins and
normalization as full-survey inference, if saved tile NPZs are unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_migration(output: Path, manifest: list[dict], shape: tuple[int, ...],
                   x_centers: np.ndarray, y_centers: np.ndarray,
                   source: Path, channel: int, use_cache: bool = True):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from src.physics.gravity_forward import OBS_X, OBS_Y
    from src.physics.gravity_migration import calculate_migration, normalize_migration

    all_cached = use_cache and all((output/'tiles'/f"{r['tile']}.npz").exists() for r in manifest)
    if all_cached:
        print('Using saved per-tile migration_input arrays.', flush=True)
        linear = nearest = None
    else:
        if not source.exists():
            raise FileNotFoundError(f'Original gravity observations not found: {source}. '
                                    'Provide --input or retain tiles using --save-tiles.')
        print('Reconstructing migration from field observations (may take several minutes)...',flush=True)
        observations = np.loadtxt(source)
        field = observations[observations[:,3] == channel]
        if not len(field):
            raise ValueError(f'No channel {channel} data found in {source}')
        linear = LinearNDInterpolator(field[:,:2],field[:,4],fill_value=np.nan)
        nearest = NearestNDInterpolator(field[:,:2],field[:,4])

    nz,ny,nx=shape
    accum=np.zeros(shape,dtype=np.float32)
    weight=np.zeros((ny,nx),dtype=np.float32)
    axis=np.maximum(np.sin(np.pi*(np.arange(64)+0.5)/64)**2,0.03).astype(np.float32)
    taper=axis[:,None]*axis[None,:]
    x_start=float(x_centers[0]-5)
    y_start=float(y_centers[0]-5)
    for idx,row in enumerate(manifest,1):
        x0=float(row['x_origin_m']);y0=float(row['y_origin_m'])
        if all_cached:
            with np.load(output/'tiles'/f"{row['tile']}.npz") as tile:
                mig=tile['migration_input'].astype(np.float32)
        else:
            gx,gy=np.meshgrid(x0+OBS_X,y0+OBS_Y)
            points=np.column_stack([gx.ravel(),gy.ravel()])
            values=linear(points)
            invalid=~np.isfinite(values)
            if invalid.any(): values[invalid]=nearest(points[invalid])
            gravity=values.reshape(81,81).astype(np.float32)
            migration=calculate_migration(gravity,component='gzz',restrict=False)
            mig,_=normalize_migration(migration)
        xo=int(round((x0-x_start)/10));yo=int(round((y0-y_start)/10))
        xe=min(nx,xo+64);ye=min(ny,yo+64)
        if xo<0 or yo<0 or xe<=xo or ye<=yo:
            raise ValueError(f'Unexpected tile coordinates for {row["tile"]}')
        w=taper[:ye-yo,:xe-xo]
        accum[:,yo:ye,xo:xe]+=mig[:,:ye-yo,:xe-xo]*w
        weight[yo:ye,xo:xe]+=w
        if idx%25==0 or idx==len(manifest): print(f'Migration tiles {idx}/{len(manifest)}',flush=True)
    result=np.zeros_like(accum)
    np.divide(accum,weight[None],out=result,where=(weight>0)[None])
    return result


def cross_section_figure(output, migration, density, valid, x, y, z, clip_percentile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib import cm

    # show XY depth-slice plus central XZ and YZ cross sections for both volumes
    j=int(np.argmin(abs(y-np.median(y))))
    i=int(np.argmin(abs(x-np.median(x))))
    k=min(8,len(z)-1)  # 85 m nominal depth
    X=(x-x[0])/1000.;Y=(y-y[0])/1000.;Z=z
    fig,axes=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    for row,(title,arr) in enumerate([('Normalized Gzz migration',migration),
                                      ('E08D predicted density (g/cm³)',density)]):
        allowed=np.abs(arr[:,valid])
        limit=float(np.percentile(allowed,clip_percentile)) if allowed.size else 1.
        limit=max(limit,1e-8)
        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        data=[arr[k],arr[:,j,:],arr[:,:,i]]
        extent=[(X[0],X[-1],Y[0],Y[-1]),(X[0],X[-1],Z[-1],Z[0]),
                (Y[0],Y[-1],Z[-1],Z[0])]
        labels=[('X (km)','Y (km)',f'Horizontal slice, depth = {Z[k]:.0f} m'),
                ('X (km)','Depth (m)','East–west vertical section'),
                ('Y (km)','Depth (m)','North–south vertical section')]
        for col,(a,e,lab) in enumerate(zip(data,extent,labels)):
            ax=axes[row,col]
            im=ax.imshow(a,cmap='RdBu_r',norm=norm,origin='upper' if col else 'lower',
                         extent=e,aspect='auto',interpolation='nearest')
            ax.set(xlabel=lab[0],ylabel=lab[1],title=f'{title}\n{lab[2]}')
            fig.colorbar(im,ax=ax,shrink=.75,pad=.01)
    fig.suptitle('Thunderbird: migration vs pretrained E08D density prediction',fontsize=15)
    path=output/'migration_vs_density_sections.png'
    fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
    print('Saved:',path)


def plotly_3d(output,migration,density,valid,x,y,z, max_points, percentile, decimate):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Sample at regular intervals so the browser remains responsive.
    xx=(x-x[0])/1000.; yy=(y-y[0])/1000.
    sx=slice(None,None,decimate);sy=slice(None,None,decimate);sz=slice(None,None,1)
    xsub=xx[sx];ysub=yy[sy];zsub=z[sz]
    masksub=valid[sy,sx]
    grids=np.meshgrid(zsub,ysub,xsub,indexing='ij')
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
                      subplot_titles=('Normalized Gzz migration','E08D predicted density (g/cm³)'),
                      horizontal_spacing=.035)
    rng=np.random.default_rng(42)
    for col,(vol,name) in enumerate([(migration,'migration'),(density,'density')],1):
        arr=vol[sz,sy,sx]
        selected_values=np.abs(arr[:,masksub])
        if not selected_values.size:
            raise ValueError('No valid survey coverage')
        cut=float(np.percentile(selected_values,percentile))
        candidates=np.argwhere((np.abs(arr)>=cut)&masksub[None,:,:] & (np.abs(arr)>1e-8))
        if len(candidates)>max_points:
            candidates=candidates[rng.choice(len(candidates),max_points,replace=False)]
        if not len(candidates):
            raise ValueError('No plotted voxels; reduce --percentile')
        zz,yyi,xxi=candidates.T
        values=arr[zz,yyi,xxi]
        vmax=max(float(np.percentile(np.abs(values),99)),1e-6)
        fig.add_trace(go.Scatter3d(
            x=xsub[xxi],y=ysub[yyi],z=zsub[zz],mode='markers',name=name,
            marker=dict(size=2,opacity=.35,color=values,colorscale='RdBu',reversescale=True,
                        cmin=-vmax,cmax=vmax,showscale=True,
                        colorbar=dict(title='Norm.' if col==1 else 'g/cm³',x=.46 if col==1 else 1.01,len=.6)),
            text=[f'{v:.3g}' for v in values],hovertemplate='X=%{x:.3f} km<br>Y=%{y:.3f} km<br>Depth=%{z:.1f} m<br>Value=%{text}<extra></extra>'
        ),row=1,col=col)
    common=dict(xaxis_title='X from survey west (km)',yaxis_title='Y from survey south (km)',
                zaxis_title='Depth (m)',zaxis=dict(autorange='reversed'),aspectmode='manual',
                aspectratio=dict(x=1,y=1,z=.45))
    fig.update_layout(scene=common,scene2=common,height=760,width=1450,
                      title='Thunderbird: full-survey migration vs E08D density (thresholded voxels)',
                      showlegend=False,margin=dict(l=0,r=0,t=75,b=0))
    path=output/'migration_vs_density_3d.html'
    fig.write_html(str(path),include_plotlyjs=True)
    print('Saved:',path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=Path('outputs/ml/field/Thunderbird_E08D'))
    p.add_argument('--input',type=Path,default=None,help='Original Tbird_gravity.dat if metadata path no longer valid')
    p.add_argument('--max-points',type=int,default=14000,help='Maximum plotted 3D points per panel')
    p.add_argument('--percentile',type=float,default=96.,help='Absolute voxel magnitude percentile for 3D plot')
    p.add_argument('--decimate',type=int,default=4,help='XY voxel spacing for plotting only; 4=40 m')
    p.add_argument('--slice-clip',type=float,default=99.,help='Symmetric color scale percentile for 2D slices')
    args=p.parse_args()
    if not 0 <= args.percentile < 100 or not 0 < args.slice_clip <= 100 or args.decimate<1 or args.max_points<1:
        p.error('Invalid percentile, slice-clip, decimate, or max-points value')
    output=args.output_root
    meta=json.loads((output/'run_metadata.json').read_text())
    if not meta.get('complete',False):
        print('WARNING: reconstruction is marked incomplete (smoke test).')
    density=np.load(output/'density_3d_gcm3.npy',mmap_mode='r')
    valid=np.load(output/'coverage_mask.npy')
    x=np.load(output/'x_cell_centers_m.npy');y=np.load(output/'y_cell_centers_m.npy');z=np.load(output/'z_cell_centers_m.npy')
    if density.shape!=(len(z),len(y),len(x)) or valid.shape!=(len(y),len(x)):
        raise ValueError('Inconsistent mosaic or coordinate dimensions')
    with (output/'tile_manifest.csv').open(newline='') as f:
        manifest=list(csv.DictReader(f))
    cached=output/'migration_normalized_mosaic.npy'
    if cached.exists():
        migration=np.load(cached,mmap_mode='r')
        if migration.shape!=density.shape: raise ValueError('Cached migration shape does not match density')
        print('Using existing normalized migration mosaic',cached)
    else:
        source=args.input or Path(meta['input'])
        migration=load_migration(output,manifest,density.shape,x,y,source,int(meta['channel']))
        np.save(cached,migration)
        print('Saved migration mosaic',cached)
    cross_section_figure(output,migration,density,valid,x,y,z,args.slice_clip)
    plotly_3d(output,migration,density,valid,x,y,z,args.max_points,args.percentile,args.decimate)
    print('Done. Open the HTML file in your local browser (copy it from CHPC first).')


if __name__=='__main__': main()
