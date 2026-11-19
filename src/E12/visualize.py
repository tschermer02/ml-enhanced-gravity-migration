"""E12 field maps + Plotly isosurfaces with physical E12 dimensions."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def output_figures(out,observed,modeled,migration,normalized,density):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    clip=max(float(np.quantile(abs(observed),.99)),1e-6)
    for ax,image,name in zip(axes,[observed,modeled,modeled-observed],
                            ['Observed Gzz','Predicted Gzz','Predicted − observed']):
        im=ax.imshow(image,origin='lower',extent=[0,8,0,8],cmap='RdBu_r',vmin=-clip,vmax=clip)
        ax.set(title=name,xlabel='X (km)',ylabel='Y (km)');fig.colorbar(im,ax=ax,label='E')
    fig.savefig(out/'gravity_observed_predicted_residual.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for row,(data,name) in enumerate(((normalized,'Normalized migration'),(density,'Density (g/cm³)'))):
        clip=max(float(np.quantile(abs(data),.99)),1e-6)
        for col,sl in enumerate((data[8],data[:,32,:],data[:,:,32])):
            extent=([0,8,0,8] if col==0 else [0,8,1,0])
            im=axes[row,col].imshow(sl,origin='lower' if col==0 else 'upper',extent=extent,
                     cmap='RdBu_r',vmin=-clip,vmax=clip,aspect='auto')
            axes[row,col].set(title=f'{name}: {("XY","XZ","YZ")[col]}',xlabel='Distance (km)',
                              ylabel='Y (km)' if col==0 else 'Depth (km)')
            fig.colorbar(im,ax=axes[row,col])
    fig.savefig(out/'migration_vs_density_sections.png',dpi=220);plt.close(fig)
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
        subplot_titles=['Gzz migration (normalized)','Predicted density contrast (g/cm³)'])
    nz,ny,nx=density.shape
    z,y,x=np.meshgrid((np.arange(24)+.5)/24,(np.arange(64)+.5)*.125,
                       (np.arange(64)+.5)*.125,indexing='ij')
    for col,(volume,label) in enumerate([(normalized,'migration'),(density,'density')],1):
        for sign,color,label2 in [(1,'Reds','positive'),(-1,'Blues','negative')]:
            v=np.maximum(volume*sign,0)
            limit=float(np.max(v))
            if limit<=.01:continue
            threshold=.15 if col==1 else .1
            if col==1: threshold=max(float(np.quantile(v,.95)),limit*.2)
            fig.add_trace(go.Isosurface(x=x.ravel(),y=y.ravel(),z=z.ravel(),value=v.ravel(),
                isomin=threshold,isomax=limit,surface_count=3,colorscale=color,
                opacity=.45,showscale=False,caps=dict(x_show=False,y_show=False,z_show=False),
                name=f'{label} {label2}',showlegend=True),row=1,col=col)
    scene=dict(xaxis_title='X (km)',yaxis_title='Y (km)',zaxis_title='Depth (km)',
        xaxis=dict(range=[0,8]),yaxis=dict(range=[0,8]),
        zaxis=dict(range=[1,0]),aspectmode='manual',aspectratio=dict(x=1,y=1,z=.45))
    fig.update_layout(scene=scene,scene2=scene,width=1500,height=750,
        title='Thunderbird E12: migration vs density | original physical scale')
    fig.write_html(str(out/'migration_vs_density_3D.html'),include_plotlyjs=True)
