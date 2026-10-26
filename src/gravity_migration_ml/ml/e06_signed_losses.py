"""Signed E06 loss & dataset: E05 objective with abs-density occupancy.

The reference E05 losses treat negative bodies as background. These implementations
preserve original weights while using |density|>=0.1 for occupancy.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import tensorflow as tf

THRESHOLD=0.1
TEMP=0.025
TVERSKY_WEIGHT=0.1
DEPTH_WEIGHT=0.02


def signed_sample(path):
    with np.load(path) as archive:
        x=np.asarray(archive['migration_input'], dtype=np.float32)
        y=np.asarray(archive['density'], dtype=np.float32)
    if x.shape!=(24,64,64) or y.shape!=(24,64,64):
        raise ValueError(f'Bad volume dimensions: {path}')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y))>1:
        raise ValueError(f'Invalid signed data: {path}')
    return x[...,None],y[...,None]


def signed_dataset(root, split, batch_size=2, limit=None, seed=42):
    paths=sorted((Path(root)/split).glob('*.npz'))
    if limit is not None: paths=paths[:limit]
    if not paths: raise FileNotFoundError(f'No files for {split} in {root}')
    def gen():
        for p in paths: yield signed_sample(p)
    sig=(tf.TensorSpec((24,64,64,1),tf.float32),tf.TensorSpec((24,64,64,1),tf.float32))
    ds=tf.data.Dataset.from_generator(gen,output_signature=sig)
    if split=='train': ds=ds.shuffle(len(paths),seed=seed,reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def signed_occupancy(values):
    dtype=values.dtype
    raw=tf.sigmoid((tf.abs(values)-tf.cast(THRESHOLD,dtype))/tf.cast(TEMP,dtype))
    baseline=tf.sigmoid(tf.cast(-THRESHOLD/TEMP,dtype))
    return tf.clip_by_value((raw-baseline)/(1-baseline),0,1)

@tf.keras.utils.register_keras_serializable(package='gravity_migration_ml')
def signed_balanced_density_mse(y_true,y_pred):
    y_true=tf.cast(y_true,y_pred.dtype)
    axes=tf.range(1,tf.rank(y_true))
    inside=tf.cast(tf.abs(y_true)>0,y_pred.dtype)
    outside=1-inside
    err=tf.square(y_pred-y_true)
    body=tf.math.divide_no_nan(tf.reduce_sum(err*inside,axis=axes),tf.reduce_sum(inside,axis=axes))
    background=tf.math.divide_no_nan(tf.reduce_sum(err*outside,axis=axes),tf.reduce_sum(outside,axis=axes))
    return tf.reduce_mean(0.5*(body+background))

@tf.keras.utils.register_keras_serializable(package='gravity_migration_ml')
def signed_tversky(y_true,y_pred):
    truth=tf.cast(tf.abs(y_true)>=THRESHOLD,y_pred.dtype)
    pred=signed_occupancy(y_pred)
    axes=tf.range(1,tf.rank(y_true))
    tp=tf.reduce_sum(truth*pred,axis=axes)
    fp=tf.reduce_sum((1-truth)*pred,axis=axes)
    fn=tf.reduce_sum(truth*(1-pred),axis=axes)
    return tf.reduce_mean(1-(tp+1e-6)/(tp+0.7*fp+0.3*fn+1e-6))

@tf.keras.utils.register_keras_serializable(package='gravity_migration_ml')
def signed_vertical_profile(y_true,y_pred):
    """Same E05 vertical profile, center and spread terms, using |rho| occupancy."""
    dtype=y_pred.dtype
    truth=tf.cast(tf.abs(y_true)>=THRESHOLD,dtype)
    pred=signed_occupancy(y_pred)
    t=tf.reduce_sum(truth,axis=(2,3,4))
    p=tf.reduce_sum(pred,axis=(2,3,4))
    t=tf.math.divide_no_nan(t,tf.reduce_sum(t,axis=1,keepdims=True))
    p=tf.math.divide_no_nan(p,tf.reduce_sum(p,axis=1,keepdims=True))
    profile=tf.reduce_sum(tf.abs(t-p),axis=1)
    nz=tf.shape(y_pred)[1]
    depth=tf.cast(tf.range(nz),dtype)[None,:]/tf.maximum(tf.cast(nz-1,dtype),1.)
    tc=tf.reduce_sum(t*depth,axis=1)
    pc=tf.reduce_sum(p*depth,axis=1)
    tv=tf.reduce_sum(t*tf.square(depth-tc[:,None]),axis=1)
    pv=tf.reduce_sum(p*tf.square(depth-pc[:,None]),axis=1)
    return tf.reduce_mean(profile+0.5*tf.abs(tc-pc)+0.5*tf.abs(tf.sqrt(tv+1e-6)-tf.sqrt(pv+1e-6)))

@tf.keras.utils.register_keras_serializable(package='gravity_migration_ml')
def signed_e06_loss(y_true,y_pred):
    return signed_balanced_density_mse(y_true,y_pred)+TVERSKY_WEIGHT*signed_tversky(y_true,y_pred)+DEPTH_WEIGHT*signed_vertical_profile(y_true,y_pred)
