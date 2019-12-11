import tensorflow as tf
from packaging import version


class ScatteringTensorFlow(tf.Module):
    def __init__(self, name):
        super(ScatteringTensorFlow, self).__init__(name=name)
        self.frontend_name = 'tensorflow'

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    @tf.Module.with_name_scope
    def __call__(self, x):
        """ This function provides the standard TensorFlow calling interface for
        the scattering computation."""
        return self.scattering(x)
