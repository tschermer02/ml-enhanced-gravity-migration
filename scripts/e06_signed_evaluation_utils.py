"""Signed-body metrics and visualizations for E06_signed."""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLD=0.1
CELL=10


def calculate_signed_metrics(truth,prediction,threshold=THRESHOLD):
    error=prediction-truth
    t=np.abs(truth)>=threshold
    p=np.abs(prediction)>=threshold
    intersection=np.count_nonzero(t&p)
    union=np.count_nonzero(t|p)
    nt=np.count_nonzero(t);npred=np.count_nonzero(p)
    def depths(mask):
        idx=np.flatnonzero(np.any(mask,axis=(1,2)))
        if not len(idx): return None
        return (int(idx[0])*CELL,(int(idx[-1])+1)*CELL)
    td=depths(t);pd=depths(p)
    if not td: raise ValueError('No true density body at threshold')
    if pd is None:
        top=bottom=truth.shape[0]*CELL
        thick=td[1]-td[0]
    else:
        top=abs(pd[0]-td[0]);bottom=abs(pd[1]-td[1]);thick=abs((pd[1]-pd[0])-(td[1]-td[0]))
    sign_agreement=float(np.mean(np.sign(prediction[t])==np.sign(truth[t]))) if nt else 0.0
    # Intersection restricted to voxels with matching recovered density polarity.
    signed_intersection=np.count_nonzero(t&p&(np.sign(truth)==np.sign(prediction)))
    return dict(
        density_mae=float(np.mean(np.abs(error))),
        density_rmse=float(np.sqrt(np.mean(error**2))),
        density_mae_inside_true_body=float(np.mean(np.abs(error[t]))) if nt else 0.0,
        body_iou=float(intersection/union) if union else 1.0,
        body_dice=float(2*intersection/(nt+npred)) if nt+npred else 1.0,
        signed_body_iou=float(signed_intersection/union) if union else 1.0,
        true_body_sign_accuracy=sign_agreement,
        body_volume_ratio=float(npred/nt) if nt else 0.0,
        top_depth_error_m=float(top),bottom_depth_error_m=float(bottom),
        thickness_error_m=float(thick),empty_prediction=not bool(npred),
    )


def save_signed_figure(density,prediction,output,sample_id):
    """Centered orthogonal slices with symmetrical color scale (-1..1)."""
    nz,ny,nx=density.shape
    occupied=np.argwhere(np.abs(density)>=THRESHOLD)
    iz,iy,ix=map(int,occupied[len(occupied)//2]) if len(occupied) else (nz//2,ny//2,nx//2)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    sections=[(density[iz],prediction[iz],f'z={iz*CELL} m'),
              (density[:,iy,:],prediction[:,iy,:],f'y={iy*CELL} m'),
              (density[:,:,ix],prediction[:,:,ix],f'x={ix*CELL} m')]
    for j,(t,p,title) in enumerate(sections):
        for i,a in enumerate((t,p)):
            im=axes[i,j].imshow(a,cmap='RdBu_r',vmin=-1,vmax=1,origin='upper' if j else 'lower')
            axes[i,j].set_title(('Truth' if i==0 else 'E06 signed prediction')+' — '+title)
    fig.colorbar(im,ax=axes,label='Density contrast (g/cm³)',shrink=0.7)
    fig.suptitle(sample_id)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150);plt.close(fig)


def plot_signed_3d(density,prediction,output,frames=24,duration=8.):
    """Red = positive contrast; blue = negative contrast. Both true/pred signed."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from PIL import Image
    if frames<2: raise ValueError('At least two frames')
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
                      subplot_titles=['True signed density','Predicted signed density'])
    for col,volume in [(1,density),(2,prediction)]:
        z,y,x=np.indices(volume.shape)
        coords={'x':((x+0.5)*CELL).ravel(),'y':((y+0.5)*CELL).ravel(),
                'z':((z+0.5)*CELL).ravel()}
        for negative in (False,True):
            magnitude=(-volume if negative else volume).ravel()
            if np.max(magnitude)<THRESHOLD: continue
            fig.add_trace(go.Isosurface(
                **coords,value=magnitude,isomin=THRESHOLD,
                isomax=float(max(THRESHOLD+1e-4,np.max(magnitude))),
                surface_count=1,opacity=0.75,showscale=False,
                colorscale=[[0,'royalblue' if negative else 'orangered'],
                            [1,'royalblue' if negative else 'orangered']],
                caps=dict(x_show=False,y_show=False,z_show=False),
                name='Negative' if negative else 'Positive',showlegend=False),row=1,col=col)
    style=dict(xaxis=dict(title='East (m)',range=[0,640]),
               yaxis=dict(title='North (m)',range=[0,640]),
               zaxis=dict(title='Depth (m)',range=[240,0]),
               aspectmode='manual',aspectratio=dict(x=1,y=1,z=.55))
    fig.update_layout(scene=style,scene2=style,width=1400,height=750,
                      title='Signed density contrast: blue < 0, red > 0')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    images=[]
    for i in range(frames):
        angle=2*math.pi*i/frames
        camera=dict(eye=dict(x=2*math.cos(angle),y=2*math.sin(angle),z=1))
        fig.update_layout(scene_camera=camera,scene2_camera=camera)
        image=Image.open(BytesIO(fig.to_image(format='png',width=1400,height=750)))
        images.append(image.convert('RGB'))
        print(f'GIF frame {i+1}/{frames}',flush=True)
    images[0].save(output,save_all=True,append_images=images[1:],
                   duration=round(duration*1000/frames),loop=0)
