"""Train isolated E12BASE--E12F 3-D U-Nets using D12 datasets.

Example: python -m src.E12.train --experiment E12F --epochs 100 --batch-size 2
No changes to common models, losses, or training code.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .config import EXPERIMENTS,HEIGHTS_VARIABLE
from .model import build_model
from .losses import loss_components
from .physics_tf import forward as tf_forward

ROOT=Path(__file__).resolve().parents[2]


def paths(root,split,limit=None):
    matches=sorted((root/split).rglob('*.npz'))
    if limit is not None: matches=matches[:limit]
    if not matches:raise FileNotFoundError(f'No samples at {root/split}')
    return matches


def dataset(root,split,batch,limit=None,seed=42):
    files=paths(root,split,limit)
    def gen():
        for f in files:
            with np.load(f) as d:
                yield (d['migration_input'].astype(np.float32)[...,None],
                       d['density'].astype(np.float32)[...,None],
                       d['gravity_clean'].astype(np.float32)[...,None],
                       np.float32(d['observation_height_m']))
    specs=(tf.TensorSpec((24,64,64,1),tf.float32),
           tf.TensorSpec((24,64,64,1),tf.float32),
           tf.TensorSpec((81,81,1),tf.float32),tf.TensorSpec((),tf.float32))
    ds=tf.data.Dataset.from_generator(gen,output_signature=specs)
    if split=='train': ds=ds.shuffle(min(len(files),128),seed=seed,reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(1)


class Trainer(tf.keras.Model):
    def __init__(self,network,settings):
        super().__init__()
        self.network=network
        self.settings=settings
        self.loss_metric=tf.keras.metrics.Mean(name='loss')
        self.huber_metric=tf.keras.metrics.Mean(name='huber')
        self.tversky_metric=tf.keras.metrics.Mean(name='tversky')
        self.depth_metric=tf.keras.metrics.Mean(name='depth')
        self.physics_metric=tf.keras.metrics.Mean(name='physics')

    @property
    def metrics(self):
        return [self.loss_metric,self.huber_metric,self.tversky_metric,
                self.depth_metric,self.physics_metric]

    def _loss(self,data,training):
        x,rho,gzz,h=data
        pred=self.network(x,training=training)
        cfg=self.settings
        weights={'balanced_huber':1.,'tversky':cfg.tversky_weight,
                 'depth':cfg.depth_weight}
        forward=None
        if cfg.physics_weight>0:
            weights['physics']=cfg.physics_weight
            forward=lambda values:tf_forward(values,h,allowed_heights=cfg.heights)
        return loss_components(rho,pred,weights=weights,signed=True,
             huber_delta=.1,gravity_observed=gzz,forward_model=forward,
             gravity_normalization=100.)

    def _update(self,parts):
        for metric,key in [(self.loss_metric,'total'),(self.huber_metric,'balanced_huber'),
                           (self.tversky_metric,'tversky'),(self.depth_metric,'depth')]:
            metric.update_state(parts[key])
        self.physics_metric.update_state(parts.get('physics',tf.constant(0.,tf.float32)))
        return {m.name:m.result() for m in self.metrics}

    def train_step(self,data):
        with tf.GradientTape() as tape: parts=self._loss(data,True)
        gradients=tape.gradient(parts['total'],self.network.trainable_variables)
        if any(g is None for g in gradients):raise RuntimeError('Missing gradient')
        self.optimizer.apply_gradients(zip(gradients,self.network.trainable_variables))
        return self._update(parts)

    def test_step(self,data):
        return self._update(self._loss(data,False))


class SaveBest(tf.keras.callbacks.Callback):
    def __init__(self,network,path):
        super().__init__();self.net=network;self.path=path;self.best=np.inf
    def on_epoch_end(self,epoch,logs=None):
        val=(logs or {}).get('val_loss',np.inf)
        if np.isfinite(val) and val<self.best:
            self.best=val;self.net.save(self.path)
            print(f'Saved best network at epoch {epoch+1}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True,choices=EXPERIMENTS)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-root',type=Path,default=None)
    args=p.parse_args()
    cfg=EXPERIMENTS[args.experiment]
    data_root=ROOT/'outputs/ml/training'/cfg.dataset/'data'
    dataset_config=json.loads((data_root/'config.json').read_text())
    if dataset_config['grid']!='e12' or tuple(dataset_config['observation_heights_m'])!=cfg.heights:
        raise ValueError('Incompatible E12 dataset geometry or height distribution')
    out=args.output_root or ROOT/'outputs/ml/training'/args.experiment
    if not out.is_absolute():out=ROOT/out
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    for name in ('checkpoints','logs'): (out/name).mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    network=build_model()
    wrapper=Trainer(network,cfg)
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-3),jit_compile=False)
    ckpt=out/'checkpoints'/'best_unet3d.keras'
    info=dict(experiment=args.experiment,dataset=cfg.dataset,grid='e12',
              geometry=dataset_config,physics_weight=cfg.physics_weight,
              tversky_weight=cfg.tversky_weight,depth_weight=cfg.depth_weight,
              seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,
              height_reference='nominal height relative to flat model top')
    (out/'logs'/'training_config.json').write_text(json.dumps(info,indent=2))
    callback=[SaveBest(network,ckpt),
              tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15),
              tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=5,min_lr=1e-6),
              tf.keras.callbacks.CSVLogger(out/'logs'/'training_history.csv')]
    wrapper.fit(dataset(data_root,'train',args.batch_size,args.limit,args.seed),
                validation_data=dataset(data_root,'validation',args.batch_size,args.limit,args.seed),
                epochs=args.epochs,callbacks=callback,verbose=2)
    print('Saved model:',ckpt)

if __name__=='__main__':main()
