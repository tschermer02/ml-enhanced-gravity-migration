"""Read-only E11D/E11E gradient diagnosis on matched D10 synthetic samples.

Run from repository root:
  python diagnose_e11_gradients.py
  python diagnose_e11_gradients.py --split validation --noise noise_00

No optimizer steps, training, checkpoint writes, or dataset modifications occur.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def summarize(vec):
    arr = np.asarray(vec, np.float64).reshape(-1)
    return float(np.sqrt(np.sum(arr * arr)))


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    norm = summarize(a) * summarize(b)
    return float(np.dot(a, b) / norm) if norm > 0 else float('nan')


def choose_sample(root, split, geometry, noise):
    base = root / split
    paths = sorted(base.rglob('*.npz'))
    paths = [p for p in paths if geometry in p.parts and (noise is None or noise in p.parts)]
    if not paths:
        raise FileNotFoundError(f'No {split} {geometry} samples under {base} with noise={noise}')
    return paths[0]


def inspect(experiment, path, checkpoint, physics_weight, normalization):
    import tensorflow as tf
    from src.data.dataset import load_sample
    from src.losses.losses import e11_loss_components

    x_np, targets_np = load_sample(path, targets='physics_clean')
    x = tf.convert_to_tensor(x_np[None, ...], tf.float32)
    t = {k: tf.convert_to_tensor(np.expand_dims(v, 0), tf.float32)
         for k, v in targets_np.items()}
    model = tf.keras.models.load_model(checkpoint, compile=False)
    bias = model.get_layer('signed_density_prediction').bias

    with tf.GradientTape(persistent=True) as tape:
        prediction = model(x, training=False)
        tape.watch(prediction)
        parts = e11_loss_components(
            t['density'], prediction,
            gravity_clean=t['gravity_clean'],
            observation_height_m=t['observation_height_m'],
            dx_m=t['dx_m'], dy_m=t['dy_m'], dz_m=t['dz_m'],
            physics_weight=physics_weight,
            physics_normalization_e=normalization,
        )
        # Calculate the physics term separately even for density-only E11D.
        # This does NOT change the weights or the checkpoint.
        if physics_weight == 0:
            from src.physics.gravity_forward_tf import gravity_gzz_forward_tf
            gzz = gravity_gzz_forward_tf(
                prediction, observation_height_m=t['observation_height_m'],
                dx_m=t['dx_m'], dy_m=t['dy_m'], dz_m=t['dz_m'])
            physics_raw = tf.reduce_mean(tf.square(
                (gzz - t['gravity_clean']) / normalization))
        else:
            physics_raw = parts['physics']
        supervised = (parts['balanced_huber'] + 0.1 * parts['tversky']
                      + 0.02 * parts['depth'])
        physics_weighted = physics_weight * physics_raw

    # Gradients with respect to the actual predicted density, not only losses.
    gs = tape.gradient(supervised, prediction)
    gp = tape.gradient(physics_raw, prediction)
    gb_supervised = tape.gradient(supervised, bias)
    gb_physics = tape.gradient(physics_raw, bias)
    del tape
    for label, tensor in [('supervised output gradient', gs),
                          ('physics output gradient', gp),
                          ('supervised bias gradient', gb_supervised),
                          ('physics bias gradient', gb_physics)]:
        if tensor is None:
            raise RuntimeError(f'Missing {label} for {experiment}')

    y = targets_np['density'][..., 0]
    pred = prediction.numpy()[0, ..., 0]
    mask = np.abs(y) >= 0.1
    s = gs.numpy()[0, ..., 0]
    p = gp.numpy()[0, ..., 0] * physics_weight
    print(f'\n{experiment} | {path}')
    print(f'  supervised loss = {float(supervised):.7g}')
    print(f'  raw physics loss = {float(physics_raw):.7g}')
    print(f'  weighted physics loss = {float(physics_weighted):.7g} (weight {physics_weight:g})')
    print(f'  true-body mean truth / prediction = {y[mask].mean():+.5f} / {pred[mask].mean():+.5f} g/cm^3')
    print(f'  predicted positive fraction in true body = {(pred[mask] > 0).mean():.1%}')
    ns, np_ = summarize(s), summarize(p)
    print(f'  output gradient L2: supervised = {ns:.7g}, weighted physics = {np_:.7g}')
    print(f'  weighted physics / supervised output-gradient ratio = {np_ / max(ns, 1e-30):.6g}')
    print(f'  output-gradient cosine (supervised vs physics) = {cosine(s, p):+.4f}'
          if physics_weight else '  output-gradient cosine not applicable (physics disabled)')
    # Negative gradient = steepest descent in output space; signs report whether
    # each term alone would locally push the TRUE-BODY region positive or negative.
    print(f'  body mean descent direction: supervised = {-float(s[mask].mean()):+.6g},'
          f' weighted physics = {-float(p[mask].mean()):+.6g}')
    bs = float(gb_supervised.numpy().reshape(-1).mean())
    bp = float(gb_physics.numpy().reshape(-1).mean()) * physics_weight
    print(f'  output-head bias descent: supervised = {-bs:+.6g}, weighted physics = {-bp:+.6g}')
    return {'supervised_norm': ns, 'physics_norm': np_,
            'supervised_bias_descent': -bs, 'physics_bias_descent': -bp,
            'sign_fraction': float((pred[mask] > 0).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('outputs/ml/training/D10/data'))
    p.add_argument('--training-root', type=Path, default=Path('outputs/ml/training'))
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    p.add_argument('--noise', default='noise_00', help='Noise folder, or none to allow all')
    p.add_argument('--normalization', type=float, default=100.0)
    p.add_argument('--physics-weight', type=float, default=0.01,
                   help='Physics weight used for E11E (default 0.01)')
    p.add_argument('--geometries', nargs='+', default=['broad_intrusion', 'salt_dome', 'compact'])
    args = p.parse_args()
    if args.physics_weight <= 0 or args.normalization <= 0:
        p.error('Physics weight and normalization must be positive')
    noisedir = None if args.noise.lower() == 'none' else args.noise
    for geo in args.geometries:
        sample = choose_sample(args.data_root, args.split, geo, noisedir)
        print(f'\n{"=" * 65}\nGEOMETRY: {geo}\n{"=" * 65}')
        for exp in ['E11D', 'E11E']:
            ckpt = args.training_root / exp / 'checkpoints' / 'best_unet3d.keras'
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
            inspect(exp, sample, ckpt, 0.0 if exp == 'E11D' else args.physics_weight,
                    args.normalization)
    print('\nDone. Models and data were not modified.')


if __name__ == '__main__':
    main()
