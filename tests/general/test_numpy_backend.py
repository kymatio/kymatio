import numpy as np
from kymatio.backend.numpy_backend import NumpyBackend

def test_reshape():
    backend = NumpyBackend

    # 1D
    x = np.random.randn(3, 5, 16)
    y = backend.reshape_input(x, signal_shape=(16,))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=1)
    assert backend.shape(x) == x.shape
    assert y.shape == (15, 1, 16)
    assert np.allclose(x, xbis)

    # 2D
    x = np.random.randn(3, 5, 16, 16)
    y = backend.reshape_input(x, signal_shape=(16, 16))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=2)
    assert backend.shape(x) == x.shape
    assert y.shape == (15, 1, 16, 16)
    assert np.allclose(x, xbis)

    # 3D
    x = np.random.randn(3, 5, 16, 16, 16)
    y = backend.reshape_input(x, signal_shape=(16, 16, 16))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=3)
    assert backend.shape(x) == x.shape
    assert y.shape == (15, 1, 16, 16, 16)
    assert np.allclose(x, xbis)
