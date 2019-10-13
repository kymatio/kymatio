import tensorflow as tf

assert(int(tf.__version__[0])>=2, 'Please upgrade TensorFlow.')

from abc import ABCMeta, abstractmethod

class ScatteringTensorflow(tf.Module, metaclass=ABCMeta):
    def __init__(self, name):
        super(ScatteringTensorflow, self).__init__(name=name)

    @abstractmethod
    def build(self):
        """ Defines elementary routines."""

    @abstractmethod
    def scattering(self, x):
        """ This function should call the functional scattering."""

