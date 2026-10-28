"""Train signed E07 U-Net on mixed geologies and noise (E06_signed objective)."""
from __future__ import annotations
# BEGIN UNIFIED TRAINING IMPORTS

from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (
    ordinary_mse, balanced_density_mse, balanced_huber_loss,
    soft_tversky_loss, vertical_profile_loss, make_loss,
)

TVERSKY_WEIGHT = 0.1
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
DENSITY_THRESHOLD = 0.1
OCCUPANCY_TEMPERATURE = 0.025
DEPTH_WEIGHT = 0.02
THRESHOLD = DENSITY_THRESHOLD
TEMP = OCCUPANCY_TEMPERATURE

e03_combined_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT})
e05_combined_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT})
signed_e06_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT}, signed=True)

def signed_balanced_density_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)

def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True, subtract_baseline=True)

def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)

# END UNIFIED TRAINING IMPORTS

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-root',default='outputs/ml/training/E07/data')
    p.add_argument('--output-root',default='outputs/ml/training/E07')
    args=p.parse_args()
    root=lambda x: Path(x) if Path(x).is_absolute() else ROOT/x
    data_root=root(args.data_root);output=root(args.output_root)
    if args.epochs<1 or args.batch_size<1 or (args.limit is not None and args.limit<1):p.error('Invalid counts')
    dirs=[output/'checkpoints',output/'logs',output/'figures']
    for d in dirs:
        if d.exists() and any(d.iterdir()):raise FileExistsError(f'Existing training outputs in {d}; use another --output-root')
    for d in dirs:d.mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    train=make_dataset(data_root,'train',batch_size=args.batch_size,limit=args.limit,seed=args.seed)
    valid=make_dataset(data_root,'validation',batch_size=args.batch_size,limit=args.limit,seed=args.seed)
    original=build_unet3d()
    head=tf.keras.layers.Conv3D(1,1,activation='tanh',name='signed_density_prediction')(original.get_layer('decoder1_conv2').output)
    model=tf.keras.Model(original.input,head,name='e07_signed_unet3d')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=signed_e06_loss,
                  metrics=[signed_balanced_density_mse,signed_tversky,signed_vertical_profile],jit_compile=False)
    model.summary()
    checkpoint=dirs[0]/'best_unet3d.keras'
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6,verbose=1),
        tf.keras.callbacks.CSVLogger(str(dirs[1]/'training_history.csv'))]
    (dirs[1]/'training_config.json').write_text(json.dumps(dict(experiment='E07',seed=args.seed,
        loss='E06_signed: balanced MSE + 0.1 signed Tversky + 0.02 depth profile',
        output='tanh [-1,1] g/cm3',epochs=args.epochs,batch_size=args.batch_size,
        training_data=str(data_root),learning_rate=1e-3,checkpoint=str(checkpoint)),indent=2))
    hist=model.fit(train,validation_data=valid,epochs=args.epochs,callbacks=callbacks,verbose=2).history
    epochs=np.arange(1,len(hist['loss'])+1)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.plot(epochs,hist['loss'],label='Training');ax.plot(epochs,hist['val_loss'],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Combined loss',title='E07: training vs validation')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(dirs[2]/'training_loss.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for name,weight in [('signed_balanced_density_mse',1),('signed_tversky',TVERSKY_WEIGHT),('signed_vertical_profile',DEPTH_WEIGHT)]:
        for prefix,style in [('', '-'),('val_','--')]:
            key=prefix+name
            if key in hist:ax.plot(epochs,np.array(hist[key])*weight,style,label=key)
    ax.set(xlabel='Epoch',ylabel='Weighted loss',title='E07: loss components');ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(dirs[2]/'loss_components.png',dpi=160);plt.close(fig)
    print(f'E07 training complete; best epoch {int(np.argmin(hist["val_loss"]))+1}; checkpoint {checkpoint}',flush=True)

if __name__=='__main__':main()
