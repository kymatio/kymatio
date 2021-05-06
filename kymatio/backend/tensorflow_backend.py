from .numpy_backend import NumpyBackend
import tensorflow as tf


class TensorFlowBackend(NumpyBackend):
    name = 'tensorflow'

    @staticmethod
    def concatenate(arrays, axis=1):
        return tf.stack(arrays, axis=axis)

    @staticmethod
    def concatenate_v2(arrays, axis=1):
        return tf.concat(arrays, axis=axis)

    @staticmethod
    def modulus(x):
        norm = tf.abs(x)

        return norm

