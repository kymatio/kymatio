import pytest
from kymatio import Scattering1D
import os
import numpy as np
import io

backends = []

from kymatio.scattering1d.backend.numpy_backend import backend
backends.append(backend)

class TestScattering1DNumpy:
    @pytest.mark.parametrize('backend', backends)
    def test_Scattering1D(self, backend):
        """
        Applies scattering on a stored signal to make sure its output agrees with
        a previously calculated version.
        """
        test_data_dir = os.path.dirname(__file__)

        with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)

        x = data['x']
        J = data['J']
        Q = data['Q']
        Sx0 = data['Sx']

        T = x.shape[-1]

        scattering = Scattering1D(J, T, Q, backend=backend, frontend='numpy')

        Sx = scattering(x)
        assert np.allclose(Sx, Sx0)

def test_subsample_fourier(random_state=42):
    rng = np.random.RandomState(random_state)
    J = 10
    # 1d signal 
    x = rng.randn(2, 2**J) + 1j * rng.randn(2, 2**J)
    x_f = np.fft.fft(x, axis=-1)

    for j in range(J + 1):
        x_f_sub = backend.subsample_fourier(x_f, 2**j)
        x_sub = np.fft.ifft(x_f_sub, axis=-1)
        assert np.allclose(x[:, ::2**j], x_sub)

    with pytest.raises(TypeError) as te:
        x_bad = x.real
        backend.subsample_fourier(x_bad, 1)
    assert "should be complex" in te.value.args[0]

def test_pad():
    N = 128
    x = np.random.rand(2, 4, N)
    
    for pad_left in range(0, N - 16, 16):
        for pad_right in [pad_left, pad_left + 16]:
            x_pad = backend.pad(x, pad_left, pad_right)
            
            # compare left reflected part of padded array with left side 
            # of original array
            for t in range(1, pad_left + 1):
                assert np.allclose(x_pad[..., pad_left - t], x[..., t])
            # compare left part of padded array with left side of 
            # original array
            for t in range(x.shape[-1]):
                assert np.allclose(x_pad[..., pad_left + t], x[..., t])
            # compare right reflected part of padded array with right side
            # of original array
            for t in range(1, pad_right + 1):
                assert np.allclose(x_pad[..., x_pad.shape[-1] - 1 - pad_right + t], x[..., x.shape[-1] - 1 - t])
            # compare right part of padded array with right side of 
            # original array
            for t in range(1, pad_right + 1):
                assert np.allclose(x_pad[..., x_pad.shape[-1] - 1 - pad_right - t], x[..., x.shape[-1] - 1 - t])

    with pytest.raises(ValueError):
        backend.pad(x, x.shape[-1], 0)

    with pytest.raises(ValueError):
        backend.pad(x, 0, x.shape[-1])

def test_unpad():
    # test unpading of a random tensor
    x = np.random.rand(8, 4)

    y = backend.unpad(x, 1, 3)

    assert y.shape == (8, 2)
    assert np.allclose(y, x[:, 1:3])

    N = 128
    x = np.random.rand(2, 4, N)

    # similar to for loop in pad test
    for pad_left in range(0, N - 16, 16):
        pad_right = pad_left + 16
        x_pad = backend.pad(x, pad_left, pad_right)
        x_unpadded = backend.unpad(x_pad, pad_left, x_pad.shape[-1] - pad_right)
        assert np.allclose(x, x_unpadded)
