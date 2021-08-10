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
            if isinstance(x, int):
                x = tf.constant(x, dtype='float32')
            else:
                x = tf.constant(x)
        return tf.math.sqrt(x)

    @classmethod
    def conj(cls, x, inplace=False):
        if inplace:
            raise Exception("TensorFlow doesn't support `out=`")
        return (tf.math.conj(x) if cls._is_complex(x) else
                x)

    @classmethod
    def reshape(cls, x, shape):
        return tf.reshape(x, shape)

    @classmethod
    def transpose(cls, x, axes):
        return tf.transpose(x, axes)

    @classmethod
    def assign_slice(cls, x, x_slc, slc):
        slc_name = type(slc).__name__
        if slc_name == 'list':
            slc = [([i] if not isinstance(i, list) else i) for i in slc]
        elif slc_name == 'range':
            slc = [[i] for i in slc]
        elif slc_name == 'slice':
            slc = [[i] for i in range(slc.start, slc.stop)]
        else:
            raise TypeError("`slc` must be list, range, or slice "
                            "(got %s)" % slc_name)

        tf.tensor_scatter_nd_update(x, slc, x_slc)
