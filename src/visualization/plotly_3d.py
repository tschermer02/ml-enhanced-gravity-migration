"""Append opt-in E11 real-coordinate 3D plot to plotly_3d.py.
Run from repository root: python update_plotly_3d.py
Legacy plot_sample_3d, save_sample_3d and GIF functions remain unchanged.
"""
from pathlib import Path
p=Path('src/visualization/plotly_3d.py')
if not p.exists():raise SystemExit('Run from repository root')
s=p.read_text()
if 'def plot_e11_regional_3d(' in s:
    print('E11 plotting functions already present');raise SystemExit(0)
addition='''

# -------- E11 anisotropic full-region density and migration visualization ---
def plot_e11_regional_3d(
    migration, density, output_path, *, x_cell_centers_m=None,
    y_cell_centers_m=None, z_cell_centers_m=None, threshold=0.1,
    migration_percentile=97.0, max_points=18000, make_gif=False,
    gif_frames=80, gif_duration_seconds=8.0,
):
    """Two-panel Plotly visualization, both panels in physical kilometers.

    Migration uses an absolute-percentile selection; density uses a fixed
    absolute density threshold (g/cm3). Actual vertical cell spacing retained.
    Output is .html or .gif depending on path/make_gif.
    """
    migration=np.asarray(migration,dtype=np.float32)
    density=np.asarray(density,dtype=np.float32)
    if migration.shape!=density.shape or migration.shape!=(24,64,64):
        raise ValueError('Expected both volumes to have shape (24,64,64)')
    if not np.isfinite(migration).all() or not np.isfinite(density).all():
        raise ValueError('Volumes contain NaN or infinity')
    if threshold<=0 or not 0<migration_percentile<100 or max_points<1:
        raise ValueError('Invalid plotting thresholds')
    nz,ny,nx=density.shape
    x=np.asarray(x_cell_centers_m if x_cell_centers_m is not None else
                 (np.arange(nx)+.5)*125,dtype=np.float64)
    y=np.asarray(y_cell_centers_m if y_cell_centers_m is not None else
                 (np.arange(ny)+.5)*125,dtype=np.float64)
    z=np.asarray(z_cell_centers_m if z_cell_centers_m is not None else
                 (np.arange(nz)+.5)*(1000/24),dtype=np.float64)
    if (x.shape!=(nx,) or y.shape!=(ny,) or z.shape!=(nz,) or
        np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0) or np.any(np.diff(z)<=0)):
        raise ValueError('Invalid coordinate vectors')
    rng=np.random.default_rng(42)
    fig=make_subplots(rows=1,cols=2,
        specs=[[{'type':'scene'},{'type':'scene'}]],
        subplot_titles=('Normalized Gzz migration','Predicted density (g/cm³)'))
    for panel, (volume, kind) in enumerate(((migration,'migration'),(density,'density')),1):
        cut=(float(np.percentile(np.abs(volume),migration_percentile))
             if kind=='migration' else threshold)
        idx=np.argwhere(np.abs(volume)>=max(cut,1e-8))
        if len(idx)>max_points:
            idx=idx[rng.choice(len(idx),max_points,replace=False)]
        if not len(idx):continue
        k,j,i=idx.T
        values=volume[k,j,i]
        limit=max(float(np.quantile(np.abs(values),.99)),1e-6)
        fig.add_trace(go.Scatter3d(
            x=(x[i]-x[0])/1000,y=(y[j]-y[0])/1000,z=z[k]/1000,
            mode='markers',name=kind,
            marker=dict(size=2.0,opacity=.35,color=values,colorscale='RdBu',
                        reversescale=True,cmin=-limit,cmax=limit,
                        showscale=True,colorbar=dict(title='Norm.' if panel==1 else 'g/cm³',
                                                      x=.46 if panel==1 else 1.02)),
            hovertemplate='X=%{x:.2f} km<br>Y=%{y:.2f} km<br>Depth=%{z:.3f} km<extra></extra>'
        ),row=1,col=panel)
    common=dict(xaxis_title='X (km)',yaxis_title='Y (km)',
                zaxis_title='Depth (km)',zaxis=dict(range=[1.,0.]),
                aspectmode='manual',aspectratio=dict(x=1,y=1,z=.35))
    fig.update_layout(scene=common,scene2=common,width=1450,height=750,
                      margin=dict(l=0,r=0,t=75,b=0),showlegend=False)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    if make_gif or output_path.suffix.lower()=='.gif':
        return save_rotation_gif(fig,output_path.with_suffix('.gif'),
            frames=gif_frames,duration_seconds=gif_duration_seconds,
            has_reconstruction=True)
    fig.write_html(str(output_path),include_plotlyjs=True)
    return output_path


def e11_plot_cli():
    import argparse
    parser=argparse.ArgumentParser(description='Plot E11 regional migration and density')
    parser.add_argument('--e11-npz',required=True,help='Thunderbird_e11_preprocessed.npz')
    parser.add_argument('--density-npz',required=True,help='NPZ containing density or prediction')
    parser.add_argument('--output',default='outputs/ml/field/Thunderbird_E11_regional/comparison_3d.html')
    parser.add_argument('--gif',action='store_true')
    args=parser.parse_args()
    with np.load(args.e11_npz) as f:
        migration=f['migration_input']
        x=f['x_cell_centers_m'];y=f['y_cell_centers_m'];z=f['z_cell_centers_m']
    with np.load(args.density_npz) as f:
        key='prediction' if 'prediction' in f else 'density'
        rho=np.asarray(f[key],dtype=np.float32).squeeze()
    path=plot_e11_regional_3d(migration,rho,args.output,
        x_cell_centers_m=x,y_cell_centers_m=y,z_cell_centers_m=z,
        make_gif=args.gif)
    print('Saved',path)

if __name__=='__main__':
    e11_plot_cli()
'''
s=s.rstrip()+addition+'\n'
compile(s,str(p),'exec');p.write_text(s);print('Updated:',p)
