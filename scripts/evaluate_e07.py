"""Evaluate E07 on 4 geology types x 4 paired gravity-noise levels.

Outputs CSV and JSON summaries by geology/noise, saved predictions, and optional
slice figures / 3D GIFs. Uses signed-density metrics from E06_signed.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/'src', ROOT/'scripts'):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
from gravity_migration_ml.ml.e06_signed_losses import signed_sample
from e06_signed_evaluation_utils import calculate_signed_metrics, save_signed_figure
from visualization.plotly_3d import plot_sample_3d

CASES=('compact','multiple','dipping','salt_dome')
LEVELS=(0,5,10,15)


def resolve(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def summary(rows):
    metrics=[k for k in rows[0] if k not in ('sample_id','case','case_index','noise_percent','empty_prediction')]
    return dict(sample_count=len(rows),empty_prediction_count=sum(bool(r['empty_prediction']) for r in rows),
                metrics={key:{'mean':float(np.mean([float(r[key]) for r in rows])),
                              'std':float(np.std([float(r[key]) for r in rows]))} for key in metrics})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='src/data/dataset_e07/manifest.csv')
    p.add_argument('--data-root',default='outputs/ml/training/E07/data')
    p.add_argument('--output-root',default='outputs/ml/predict/E07')
    p.add_argument('--model',default='outputs/ml/training/E07/checkpoints/best_unet3d.keras')
    p.add_argument('--visualize-count',type=int,default=2)
    p.add_argument('--plotly',action='store_true')
    p.add_argument('--gif-frames',type=int,default=24)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.visualize_count<0 or args.gif_frames<2:p.error('Invalid visualization options')
    manifest,data_root,output_root,model_path=map(resolve,(args.manifest,args.data_root,args.output_root,args.model))
    with manifest.open(newline='') as f:records=[r for r in csv.DictReader(f) if r['split']=='test']
    if not records:raise ValueError('No test samples')
    if (output_root/'aggregate_summary.json').exists() and not args.overwrite:
        raise FileExistsError(f'{output_root} already evaluated; supply --overwrite')
    model=tf.keras.models.load_model(model_path,compile=False)
    records.sort(key=lambda r:(r['case'],int(r['noise_percent']),int(r['case_index'])))
    rows=[];indices=defaultdict(int)
    for record in records:
        case=record['case'];noise=int(record['noise_percent']);sid=record['sample_id']
        x,y=signed_sample(data_root/'test'/(sid+'.npz'))
        truth=y[...,0]
        pred=np.asarray(model(x[None],training=False).numpy()[0,...,0],dtype=np.float32)
        metrics=calculate_signed_metrics(truth,pred)
        result=dict(sample_id=sid,case=case,case_index=int(record['case_index']),noise_percent=noise,**metrics)
        rows.append(result)
        folder=output_root/case/f'noise_{noise:02d}'
        pred_dir=folder/'predictions'/'test'; pred_dir.mkdir(parents=True,exist_ok=True)
        with np.load(data_root/'test'/(sid+'.npz')) as f:
            migration=np.asarray(f['migration'],dtype=np.float32)
            scale=float(f['migration_scale'])
        np.savez_compressed(pred_dir/(sid+'.npz'),density=truth,prediction=pred,
                            migration=migration,migration_input=x[...,0],migration_scale=scale)
        indices[(case,noise)]+=1
        if indices[(case,noise)]<=args.visualize_count:
            figures=folder/'figures'/'test';figures.mkdir(parents=True,exist_ok=True)
            save_signed_figure(truth,pred,figures/(sid+'_density_comparison.png'),sid)
            if args.plotly:
                plot_sample_3d(truth,None,figures/(sid+'_3d.gif'),reconstruction=pred,
                               gif_frames=args.gif_frames,make_gif=True)
        if len(rows)%20==0:print(f'Evaluated {len(rows)}/{len(records)}',flush=True)
    output_root.mkdir(parents=True,exist_ok=True)
    with (output_root/'all_test_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    category={}
    for case in CASES:
        category[case]={}
        for noise in LEVELS:
            subset=[r for r in rows if r['case']==case and r['noise_percent']==noise]
            if not subset:continue
            category[case][f'noise_{noise:02d}']=summary(subset)
            folder=output_root/case/f'noise_{noise:02d}'/'predictions'
            (folder/'test_summary.json').write_text(json.dumps(category[case][f'noise_{noise:02d}'],indent=2))
    result=dict(experiment='E07',sample_count=len(rows),geologies=list(CASES),noise_percent=list(LEVELS),
                overall=summary(rows),per_category_noise=category)
    (output_root/'aggregate_summary.json').write_text(json.dumps(result,indent=2))
    print(f'E07 evaluation complete: {len(rows)} samples; {output_root}',flush=True)

if __name__=='__main__':main()
