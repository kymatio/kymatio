import tensorflow as tf
from packaging import version

assert version.parse(tf.__version__)>=version.parse("2.0.0a0"), 'Current TensorFlow version is '+str(tf.__version__)+\
                                                                ' . Please upgrade TensorFlow to 2.0.0a0 at least.'

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

