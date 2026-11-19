"""Independent E12 synthetic prediction, density metrics, Gzz residuals.
Run: python -m src.E12.evaluate --experiment E12F --split test
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from .config import EXPERIMENTS
from .physics import forward
ROOT=Path(__file__).resolve().parents[2]

def load_model(exp):
    import tensorflow as tf
    p=ROOT/'outputs/ml/training'/exp/'checkpoints/best_unet3d.keras'
    return tf.keras.models.load_model(p,compile=False)

def metrics(y,p,obs,sim,threshold=.1):
    a=np.abs(y)>=threshold;b=np.abs(p)>=threshold
    inter=np.count_nonzero(a&b);union=np.count_nonzero(a|b)
    dice=2*inter/max(np.count_nonzero(a)+np.count_nonzero(b),1)
    signed_inter=np.count_nonzero(a&b&(np.sign(y)==np.sign(p)))
    # Signed IoU analogous to class-specific union for positive+negative classes.
    signed_union=0
    for sign in [1,-1]:
        signed_union+=np.count_nonzero(((y*sign)>=threshold)|((p*sign)>=threshold))
    truth_body=a
    sign_correct=float(np.mean(np.sign(y[truth_body])==np.sign(p[truth_body]))) if a.any() else np.nan
    return dict(iou=inter/max(union,1),dice=dice,signed_iou=signed_inter/max(signed_union,1),
                sign_accuracy=sign_correct,density_rmse=float(np.sqrt(np.mean((p-y)**2))),
                density_mae_inside_body=float(np.mean(abs(p[a]-y[a]))) if a.any() else np.nan,
                gzz_rmse_e=float(np.sqrt(np.mean((sim-obs)**2))),
                gzz_correlation=float(np.corrcoef(obs.ravel(),sim.ravel())[0,1])
                if np.std(sim)>1e-8 and np.std(obs)>1e-8 else np.nan)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True,choices=EXPERIMENTS)
    p.add_argument('--split',choices=('validation','test'),default='test')
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    data_root=ROOT/'outputs/ml/training'/EXPERIMENTS[args.experiment].dataset/'data'/args.split
    paths=sorted(data_root.rglob('*.npz'))
    if args.limit:paths=paths[:args.limit]
    if not paths:raise FileNotFoundError(data_root)
    output=ROOT/'outputs/ml/predict'/args.experiment
    dest=output/f'{args.split}_metrics.csv'
    if dest.exists() and not args.overwrite:raise FileExistsError(dest)
    output.mkdir(parents=True,exist_ok=True)
    model=load_model(args.experiment)
    rows=[]
    for n,path in enumerate(paths):
        with np.load(path) as d:
            x=np.asarray(d['migration_input'],np.float32)
            y=np.asarray(d['density'],np.float32)
            obs=np.asarray(d['gravity'],np.float32)
            h=float(d['observation_height_m'])
            geom=str(d['geometry'])
        pred=np.asarray(model(x[None,...,None],training=False).numpy()[0,...,0])
        sim=forward(pred,h)
        row=dict(filename=str(path.relative_to(data_root)),geometry=geom,
                 noise_percent=float(path.parent.name.removeprefix('noise_')),
                 observation_height_m=h,**metrics(y,pred,obs,sim))
        rows.append(row)
        if (n+1)%25==0 or n+1==len(paths):print(f'{args.experiment}: {n+1}/{len(paths)}',flush=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=dict(experiment=args.experiment,dataset=EXPERIMENTS[args.experiment].dataset,
                 n_samples=len(rows),overall={},by_geometry={})
    fields=[k for k in rows[0] if k not in ('filename','geometry','noise_percent','observation_height_m')]
    def summarize(items):return {k:float(np.nanmean([r[k] for r in items])) for k in fields}
    summary['overall']=summarize(rows)
    summary['by_geometry']={g:summarize([r for r in rows if r['geometry']==g]) for g in sorted(set(r['geometry'] for r in rows))}
    result=output/f'{args.split}_summary.json'
    result.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Saved',dest,result)
if __name__=='__main__':main()
