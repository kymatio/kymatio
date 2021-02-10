from .numpy_backend import NumpyBackend
import tensorflow as tf


class TensorFlowBackend(NumpyBackend):
    name = 'tensorflow'

    @staticmethod
    def concatenate(arrays):
        return tf.stack(arrays, axis=1)

    @staticmethod
    def modulus(x):
        norm = tf.abs(x)

        return norm

