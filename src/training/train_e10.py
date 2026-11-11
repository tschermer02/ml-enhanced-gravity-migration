"""Train E10A (per-sample normalized migration) or E10B (raw migration).

Both use exactly the same regional dataset and E08D model/loss.
Example:
 python -m src.training.train_e10 --name E10A --data-root /scratch/general/vast/$USER/e10/data --output-root /scratch/general/vast/$USER/e10/E10A
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from src.data.e10_regional import SHAPE
from src.models.unet3d import build_unet3d
from src.losses.losses import (make_loss,balanced_huber_loss,
                               balanced_density_mse,soft_tversky_loss,vertical_profile_loss)


def load_pair(path, normalize):
    with np.load(path,allow_pickle=False) as f:
        density=np.asarray(f['density'],np.float32)
        migration=np.asarray(f['migration'],np.float32)
    if density.shape!=SHAPE or migration.shape!=SHAPE:
        raise ValueError(f'Wrong shape in {path}')
    if normalize:
        migration=migration/max(float(np.max(np.abs(migration))),1e-8)
    if not np.isfinite(migration).all(): raise ValueError(f'Nonfinite migration: {path}')
    return migration[...,None],density[...,None]


def dataset(paths,batch,normalize,shuffle,seed):
    paths=tuple(map(str,paths))
    spec=tf.TensorSpec((*SHAPE,1),tf.float32)
    def gen():
        for path in paths: yield load_pair(path,normalize)
    d=tf.data.Dataset.from_generator(gen,output_signature=(spec,spec))
    if shuffle: d=d.shuffle(len(paths),seed=seed,reshuffle_each_iteration=True)
    # Large tensors; avoid large-memory prefetch.
    return d.batch(batch).prefetch(1)


def signed_balanced_huber(y_true, y_pred):
    return balanced_huber_loss(y_true, y_pred, delta=.1, signed=True)


def signed_balanced_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',choices=['E10A','E10B'],required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.batch_size<1 or args.epochs<1:p.error('epochs and batch-size must be positive')
    data_root=args.data_root.expanduser().resolve()
    out=args.output_root.expanduser().resolve()
    normalize=args.name=='E10A'
    paths={split:sorted((data_root/split).rglob('*.npz')) for split in ('train','validation','test')}
    if not all(paths.values()): raise FileNotFoundError('Dataset splits missing under '+str(data_root))
    if args.limit: paths={k:v[:args.limit] for k,v in paths.items()}
    if out.exists() and any(out.rglob('*')): raise FileExistsError(f'Output not empty: {out}')
    (out/'checkpoints').mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir(parents=True,exist_ok=True)
    # Same initialization, optimizer, batches, sample order and loss for A/B.
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    train=dataset(paths['train'],args.batch_size,normalize,True,args.seed)
    valid=dataset(paths['validation'],args.batch_size,normalize,False,args.seed)
    base=build_unet3d(input_shape=(*SHAPE,1))
    output=tf.keras.layers.Conv3D(1,1,activation='tanh',name='signed_density_prediction')(
        base.get_layer('decoder1_conv2').output)
    model=tf.keras.Model(base.input,output,name='e10_e08d_unet3d')
    loss=make_loss(weights={'balanced_huber':1.0,'tversky':.1,'depth':.02},
                   signed=True,huber_delta=.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=loss,
                  metrics=[signed_balanced_huber,signed_balanced_mse],
                  jit_compile=False)
    ckpt=out/'checkpoints'/'best_unet3d.keras'
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(ckpt),monitor='val_loss',save_best_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=5,factor=.5,min_lr=1e-6),
               tf.keras.callbacks.CSVLogger(str(out/'logs'/'training_history.csv'))]
    info={'name':args.name,'data_root':str(data_root),'normalized_per_sample':normalize,
          'input':'migration/max(maxabs,1e-8)' if normalize else 'raw migration, unscaled',
          'architecture':'E08D U-Net 16/32/64, signed tanh output',
          'loss':'balanced Huber delta=.1 + .1 Tversky + .02 vertical profile',
          'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,
          'train_count':len(paths['train']),'validation_count':len(paths['validation']),
          'test_count':len(paths['test'])}
    (out/'logs'/'training_config.json').write_text(json.dumps(info,indent=2))
    model.summary()
    history=model.fit(train,validation_data=valid,epochs=args.epochs,callbacks=callbacks,verbose=2)
    if not np.isfinite(np.asarray(history.history['loss'])).all():
        raise RuntimeError('Non-finite training loss: E10B raw input may need a separate globally scaled baseline')
    print('Best checkpoint:',ckpt,flush=True)
    # Independent held-out test, checkpoint weights from EarlyStopping (or best checkpoint).
    model=tf.keras.models.load_model(ckpt,compile=False)
    rows=[]
    for i,path in enumerate(paths['test'],1):
        x,y=load_pair(path,normalize)
        prediction=np.asarray(model(x[None,...],training=False))[0,...,0]
        truth=y[...,0]
        occupied_true=np.abs(truth)>=0.1
        occupied_pred=np.abs(prediction)>=0.1
        inter=np.count_nonzero(occupied_true & occupied_pred)
        union=np.count_nonzero(occupied_true | occupied_pred)
        denom=np.count_nonzero(occupied_true)+np.count_nonzero(occupied_pred)
        rows.append({'sample':path.name,'geometry':path.parent.name,
                     'iou':inter/union if union else 1.,
                     'dice':2*inter/denom if denom else 1.,
                     'mae':float(np.mean(np.abs(truth-prediction)))})
        if i%25==0: print('Evaluated',i,'/',len(paths['test']),flush=True)
    with (out/'logs'/'test_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample','geometry','iou','dice','mae']);w.writeheader();w.writerows(rows)
    summary={g:{metric:float(np.mean([r[metric] for r in rows if r['geometry']==g]))
                for metric in ('iou','dice','mae')}
             for g in sorted({r['geometry'] for r in rows})}
    summary['overall']={metric:float(np.mean([r[metric] for r in rows])) for metric in ('iou','dice','mae')}
    (out/'logs'/'test_summary.json').write_text(json.dumps(summary,indent=2))
    print('Test summary:',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
