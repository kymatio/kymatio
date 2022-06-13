from .numpy_backend import NumpyBackend
import tensorflow as tf


class TensorFlowBackend(NumpyBackend):
    name = 'tensorflow'

    @staticmethod
    def concatenate(arrays, dim=1):
        return tf.stack(arrays, axis=dim)

    @staticmethod
    def modulus(x):
        norm = tf.abs(x)

        return norm

    @staticmethod
    def reshape_input(x, signal_shape, n_inserted_dims=0):
        new_shape = tf.concat(((-1,) + (1,)*n_inserted_dims, signal_shape), 0)
        return tf.reshape(x, new_shape)

    @staticmethod
    def reshape_output(S, batch_shape, n_inserted_dims, n_kept_dims):
        new_shape = tf.concat(
            (batch_shape, (1,)*n_inserted_dims, S.shape[-n_kept_dims:]), 0)
        return tf.reshape(S, new_shape)

    @staticmethod
    def shape(x):
        return tf.shape(x)
