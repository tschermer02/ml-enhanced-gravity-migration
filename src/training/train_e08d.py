"""E08D: D06C Gzz signed-density U-Net, balanced Huber versus E08C MSE.

Usage:
    python -m src.training.train_e08d --epochs 100 --batch-size 2 \
        --data-root outputs/ml/training/D06C/data \
        --output-root outputs/ml/training/E08D
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (
    balanced_huber_loss, balanced_density_mse,
    soft_tversky_loss, vertical_profile_loss, make_loss,
)

ROOT = Path(__file__).resolve().parents[2]
TVERSKY_WEIGHT = 0.1
DEPTH_WEIGHT = 0.02
HUBER_DELTA = 0.1  # g/cm^3
LEARNING_RATE = 1e-3

signed_e08d_loss = make_loss(
    weights={'balanced_huber': 1.0, 'tversky': TVERSKY_WEIGHT, 'depth': DEPTH_WEIGHT},
    signed=True, huber_delta=HUBER_DELTA,
)

def signed_balanced_huber(y_true, y_pred):
    return balanced_huber_loss(y_true, y_pred, delta=HUBER_DELTA, signed=True)

def signed_balanced_density_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)

def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True, subtract_baseline=True)

def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)

def plot_history(hist, figures):
    epochs = np.arange(1, len(hist['loss'])+1)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.plot(epochs,hist['loss'],label='Training')
    ax.plot(epochs,hist['val_loss'],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Combined loss',title='E08D: training vs validation')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(figures/'training_loss.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for name,weight in [('signed_balanced_huber',1),('signed_tversky',TVERSKY_WEIGHT),('signed_vertical_profile',DEPTH_WEIGHT)]:
        for prefix,style in [('', '-'),('val_','--')]:
            key=prefix+name
            if key in hist:ax.plot(epochs,np.asarray(hist[key])*weight,style,label=key)
    ax.set(xlabel='Epoch',ylabel='Weighted loss',title='E08D: loss components')
    ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(figures/'loss_components.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-root',default='outputs/ml/training/D06C/data')
    p.add_argument('--output-root',default='outputs/ml/training/E08D')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or (args.limit is not None and args.limit<1):
        p.error('epochs, batch-size and limit must be positive')
    resolve=lambda x: Path(x) if Path(x).is_absolute() else ROOT/x
    data_root,output=resolve(args.data_root),resolve(args.output_root)
    dirs=[output/'checkpoints',output/'logs',output/'figures']
    for d in dirs:
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f'Existing training outputs in {d}; use another --output-root')
    train_count=len(sample_paths(data_root,'train',args.limit))
    val_count=len(sample_paths(data_root,'validation',args.limit))
    for d in dirs:d.mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    train=make_dataset(data_root,'train',batch_size=args.batch_size,limit=args.limit,seed=args.seed)
    valid=make_dataset(data_root,'validation',batch_size=args.batch_size,limit=args.limit,seed=args.seed)
    original=build_unet3d()
    head=tf.keras.layers.Conv3D(1,1,activation='tanh',name='signed_density_prediction')(
        original.get_layer('decoder1_conv2').output)
    model=tf.keras.Model(original.input,head,name='e07_signed_unet3d')
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),loss=signed_e08d_loss,
                  metrics=[signed_balanced_huber,signed_balanced_density_mse,
                           signed_tversky,signed_vertical_profile],jit_compile=False)
    model.summary()
    checkpoint=dirs[0]/'best_unet3d.keras'
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=5,min_lr=1e-6,verbose=1),
        tf.keras.callbacks.CSVLogger(str(dirs[1]/'training_history.csv')),
    ]
    (dirs[1]/'training_config.json').write_text(json.dumps(dict(
        experiment='E08D',comparison_experiment='E08C',seed=args.seed,
        loss='signed balanced Huber (delta=0.1) + 0.1 signed Tversky + 0.02 depth profile',
        huber_delta_gcm3=HUBER_DELTA,output='tanh [-1,1] g/cm3',
        epochs=args.epochs,batch_size=args.batch_size,training_data=str(data_root),
        learning_rate=LEARNING_RATE,checkpoint=str(checkpoint),
        train_count=train_count,validation_count=val_count),indent=2))
    hist=model.fit(train,validation_data=valid,epochs=args.epochs,callbacks=callbacks,verbose=2).history
    plot_history(hist,dirs[2])
    print(f'E08D training complete; best epoch {int(np.argmin(hist["val_loss"]))+1}; checkpoint {checkpoint}',flush=True)

if __name__=='__main__':main()
