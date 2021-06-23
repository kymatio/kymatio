from .numpy_backend import NumpyBackend
import tensorflow as tf


class TensorFlowBackend(NumpyBackend):
    name = 'tensorflow'

    @staticmethod
    def concatenate(arrays, axis=-2):
        return tf.stack(arrays, axis=axis)

    @staticmethod
    def concatenate_v2(arrays, axis=1):
        return tf.concat(arrays, axis=axis)

    @staticmethod
    def modulus(x):
        norm = tf.abs(x)

        return norm

    @classmethod
    def sqrt(cls, x):
        if isinstance(x, (int, float)):
            x = tf.constant(x)
        return tf.math.sqrt(x)

    @classmethod
    def conj(cls, x, inplace=False):
        if inplace:
            raise Exception("TensorFlow doesn't support `out=`")
        return (tf.math.conj(x) if cls._is_complex(x) else
                x)
