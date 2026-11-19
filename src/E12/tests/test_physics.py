"""Run from repo root: python -m unittest src.E12.tests.test_physics -v
TF checks only run when TensorFlow is installed (e.g., CHPC gravity-tf).
"""
import unittest
import numpy as np
from src.E12.config import EXPERIMENTS,OBS_SHAPE,SHAPE
from src.E12.physics import forward,direct_forward,migration,normalize_migration

class E12PhysicsTests(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(SHAPE,(24,64,64));self.assertEqual(OBS_SHAPE,(81,81))
        self.assertEqual(len(EXPERIMENTS),7)

    def test_direct_vs_fft_both_signs_heights(self):
        for h in (0.,100.,200.):
            rho=np.zeros(SHAPE,np.float32);rho[1,8,20]=.55;rho[10,35,40]=-.3
            fft=forward(rho,h);ref=direct_forward(rho,h)
            np.testing.assert_allclose(fft,ref,atol=1e-3,rtol=1e-4)

    def test_adjoint(self):
        rng=np.random.default_rng(17)
        for h in (0.,100.):
            rho=np.zeros(SHAPE,np.float32)
            rho[2,10,20]=.5;rho[14,31,44]=-.2
            g=rng.normal(size=OBS_SHAPE).astype(np.float32)
            lhs=np.vdot(forward(rho,h),g)
            rhs=np.vdot(rho,migration(g,h,depth_weight=False))
            np.testing.assert_allclose(lhs,rhs,atol=.001,rtol=1e-4)

    def test_normalize(self):
        rho=np.zeros(SHAPE,np.float32);rho[2,10,20]=1
        normalized,scale=normalize_migration(rho)
        self.assertEqual(scale,1.0)
        self.assertEqual(float(normalized.max()),1)

    @unittest.skipUnless(__import__('importlib').util.find_spec('tensorflow') is not None,
                         'TensorFlow not installed (run on gravity-tf)')
    def test_tensorflow_forward_and_gradient(self):
        import tensorflow as tf
        from src.E12.physics_tf import forward as tf_forward
        rho=np.zeros(SHAPE,np.float32)
        rho[2,10,20]=.5;rho[9,50,13]=-.4
        batch=tf.Variable(rho[None,...,None])
        for height in (0.,100.,200.):
            with tf.GradientTape() as tape:
                response=tf_forward(batch,tf.constant([height],tf.float32))
                val=tf.reduce_sum(response**2)
            grad=tape.gradient(val,batch)
            self.assertIsNotNone(grad)
            self.assertTrue(np.isfinite(grad.numpy()).all())
            np.testing.assert_allclose(response.numpy()[0,...,0],forward(rho,height),atol=1e-3,rtol=1e-4)
        # Directional derivative finite difference, deterministic nonzero cell.
        h=100.
        with tf.GradientTape() as tape:
            y=tf_forward(batch,tf.constant([h]))
            loss=tf.reduce_mean(y*y)
        g=tape.gradient(loss,batch).numpy()[0,2,10,20,0]
        eps=1e-3
        positive=rho.copy();negative=rho.copy()
        positive[2,10,20]+=eps;negative[2,10,20]-=eps
        fd=(np.mean(forward(positive,h)**2)-np.mean(forward(negative,h)**2))/(2*eps)
        np.testing.assert_allclose(g,fd,rtol=.03,atol=1e-4)

if __name__=='__main__':unittest.main()
