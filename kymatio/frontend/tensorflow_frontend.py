import tensorflow as tf
from packaging import version
<<<<<<< HEAD
assert version.parse(tf.__version__)>=version.parse("2.0.0a0"), 'Current TensorFlow version is '+str(tf.__version__)+\
                                                                ' . Please upgrade TensorFlow to 2.0.0a0 at least.'

class ScatteringTensorFlow(tf.Module):
    def __init__(self, name):
        super(ScatteringTensorFlow, self).__init__(name=name)

    def build(self):
        """ Defines elementary routines."""
        raise NotImplementedError

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    def loginfo(self):
        """ Returns the logging message when the frontend is deployed."""
        raise NotImplementedError
       
=======

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

>>>>>>> e801b80... adding all basic files
