"""E10C: E08D trained on the same D06C data, with raw migration input.

Run from repository root:
python -m src.training.train_e10c \
 --data-root /scratch/general/vast/$USER/gravity_datasets/D06C/data \
 --output-root outputs/ml/training/E10C --epochs 100 --batch-size 2
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
from src.data.dataset import sample_paths, VOLUME_SHAPE
from src.models.unet3d import build_unet3d
from src.losses.losses import (make_loss, balanced_huber_loss, balanced_density_mse,
                               soft_tversky_loss, vertical_profile_loss)

ROOT = Path(__file__).resolve().parents[2]
TVERSKY_WEIGHT = .1
DEPTH_WEIGHT = .02
HUBER_DELTA = .1
LEARNING_RATE = 1e-3


def load_raw_pair(path):
    with np.load(path, allow_pickle=False) as archive:
        # IMPORTANT: unlike E08D, do not use archive['migration_input'].
        migration = np.asarray(archive['migration'], dtype=np.float32)
        density = np.asarray(archive['density'], dtype=np.float32)
    if migration.shape != VOLUME_SHAPE or density.shape != VOLUME_SHAPE:
        raise ValueError(f'{path}: invalid migration/density shapes {migration.shape}, {density.shape}')
    if not np.isfinite(migration).all() or not np.isfinite(density).all():
        raise ValueError(f'{path}: nonfinite migration/density')
    return migration[..., None], density[..., None]


def make_raw_dataset(paths, batch_size, seed, shuffle):
    paths = tuple(map(str, paths))
    spec = tf.TensorSpec((*VOLUME_SHAPE, 1), tf.float32)
    def samples():
        for path in paths:
            yield load_raw_pair(path)
    ds = tf.data.Dataset.from_generator(samples, output_signature=(spec, spec))
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(paths)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(1)


def signed_balanced_huber(y_true, y_pred):
    return balanced_huber_loss(y_true, y_pred, delta=HUBER_DELTA, signed=True)


def signed_balanced_density_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)


def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True, subtract_baseline=True)


def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)


def plot_history(hist, figures):
    epochs = np.arange(1, len(hist['loss']) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, hist['loss'], label='Training')
    ax.plot(epochs, hist['val_loss'], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Combined loss', title='E10C: training vs validation')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(figures/'training_loss.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, weight in [('signed_balanced_huber',1), ('signed_tversky',TVERSKY_WEIGHT),
                         ('signed_vertical_profile',DEPTH_WEIGHT)]:
        for prefix, style in [('', '-'), ('val_', '--')]:
            key = prefix + name
            if key in hist:
                ax.plot(epochs, np.asarray(hist[key])*weight, style, label=key)
    ax.set(xlabel='Epoch', ylabel='Weighted loss', title='E10C: loss components')
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(figures/'loss_components.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('outputs/ml/training/E10C'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error('epochs, batch-size and limit must be positive')
    data_root = args.data_root.expanduser().resolve()
    out = args.output_root.expanduser()
    if not out.is_absolute(): out = ROOT/out
    paths = {s: sample_paths(data_root, s, args.limit) for s in ('train','validation')}
    for split in paths:
        x, _ = load_raw_pair(paths[split][0])
        print(f'{split}: {len(paths[split])} samples; raw migration range '
              f'[{x.min():.6g},{x.max():.6g}]', flush=True)
    dirs = [out/'checkpoints', out/'logs', out/'figures']
    for d in dirs:
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f'Existing training outputs in {d}; choose a new output-root')
    for d in dirs: d.mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    train = make_raw_dataset(paths['train'], args.batch_size, args.seed, True)
    valid = make_raw_dataset(paths['validation'], args.batch_size, args.seed, False)
    original = build_unet3d()
    head = tf.keras.layers.Conv3D(1, 1, activation='tanh', name='signed_density_prediction')(
        original.get_layer('decoder1_conv2').output)
    model = tf.keras.Model(original.input, head, name='e07_signed_unet3d')
    loss = make_loss(weights={'balanced_huber':1., 'tversky':TVERSKY_WEIGHT,
                              'depth':DEPTH_WEIGHT}, signed=True, huber_delta=HUBER_DELTA)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=loss,
                  metrics=[signed_balanced_huber, signed_balanced_density_mse,
                           signed_tversky, signed_vertical_profile],jit_compile=False)
    checkpoint = dirs[0]/'best_unet3d.keras'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss',
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5,
                                             patience=5,min_lr=1e-6,verbose=1),
        tf.keras.callbacks.CSVLogger(str(dirs[1]/'training_history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    (dirs[1]/'training_config.json').write_text(json.dumps(dict(
        experiment='E10C', comparison_experiment='E08D', seed=args.seed,
        input='raw migration from D06C migration key; NO normalization or clipping',
        loss='signed balanced Huber (delta=0.1) + 0.1 signed Tversky + 0.02 depth profile',
        output='tanh [-1,1] g/cm3', epochs=args.epochs,batch_size=args.batch_size,
        training_data=str(data_root),learning_rate=LEARNING_RATE,
        checkpoint=str(checkpoint),train_count=len(paths['train']),
        validation_count=len(paths['validation'])),indent=2))
    model.summary()
    hist = model.fit(train,validation_data=valid,epochs=args.epochs,
                     callbacks=callbacks,verbose=2).history
    plot_history(hist,dirs[2])
    print(f'E10C training complete; best epoch {np.argmin(hist["val_loss"])+1}; '
          f'checkpoint {checkpoint}',flush=True)

if __name__=='__main__':
    main()
