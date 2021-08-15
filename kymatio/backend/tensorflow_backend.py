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
    def sqrt(cls, x, dtype=None):
        if isinstance(x, (int, float)):
            x = tf.constant(x, dtype=dtype)
        elif dtype is not None:
            x = tf.cast(x, dtype=dtype)
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
        """Implemented only for indexing into last axis."""
        slc_name = type(slc).__name__
        if slc_name == 'tuple':
            # handle input from `conj_reflections`
            slc = slc[-1]
            slc_name = type(slc).__name__

        if slc_name == 'list':
            pass
        elif slc_name == 'range':
            slc = list(slc)
        elif slc_name == 'slice':
            slc = list(range(slc.start or 0, slc.stop or x.shape[-1],
                             slc.step or 1))
        else:
            raise TypeError("`slc` must be list, range, or slice "
                            "(got %s)" % slc_name)

        slc_tf = tf.ones(x_slc.shape)
        slc_tf = tf.where(slc_tf).numpy().reshape([*x_slc.shape, x_slc.ndim])
        slc_tf[..., -1] = slc

        x = tf.tensor_scatter_nd_update(x, slc_tf, x_slc)
        return x
