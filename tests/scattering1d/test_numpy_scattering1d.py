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
