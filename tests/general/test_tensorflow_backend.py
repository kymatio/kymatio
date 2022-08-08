import numpy as np
import tensorflow as tf
from kymatio.backend.tensorflow_backend import TensorFlowBackend

def test_reshape():
    backend = TensorFlowBackend

    # 1D
    x = tf.random.normal((3, 5, 16))
    y = backend.reshape_input(x, signal_shape=(16,))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=1)
    assert tuple(backend.shape(x)) == tuple(x.shape)
    assert tuple(y.shape) == (15, 1, 16)
    assert np.allclose(x.numpy(), xbis.numpy())

    # 2D
    x = tf.random.normal((3, 5, 16, 16))
    y = backend.reshape_input(x, signal_shape=(16, 16))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=2)
    assert tuple(backend.shape(x)) == tuple(x.shape)
    assert tuple(y.shape) == (15, 1, 16, 16)
    assert np.allclose(x.numpy(), xbis.numpy())

    # 3D
    x = tf.random.normal((3, 5, 16, 16, 16))
    y = backend.reshape_input(x, signal_shape=(16, 16, 16))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=3)
    assert tuple(backend.shape(x)) == tuple(x.shape)
    assert tuple(y.shape) == (15, 1, 16, 16, 16)
    assert np.allclose(x.numpy(), xbis.numpy())
