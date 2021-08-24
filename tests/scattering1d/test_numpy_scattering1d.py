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

    @pytest.mark.parametrize('backend', backends)
    def test_pad_mode(self, backend):
        """Ensures `pad_mode` works as intended."""
        N = 511
        J = 6
        Q = 4
        x = np.random.randn(N)

        for mode in ('reflect', 'zero'):
            def pad_fn(x, pad_left, pad_right):
                pad_shape = (x.ndim - 1) * ((0, 0),) + ((pad_left, pad_right),)
                return np.pad(x, pad_shape,
                              mode=('reflect' if mode == 'reflect' else
                                    'constant'))
            Scxs = []
            for pad_mode in (mode, pad_fn):
                scattering = Scattering1D(J, N, Q, pad_mode=pad_mode,
                                          backend=backend, frontend='numpy')
                Scxs.append(scattering(x))
            assert np.allclose(Scxs[0], Scxs[1])

        with pytest.raises(ValueError) as record:
            _ = Scattering1D(
                J, N, Q, backend=backend, frontend='numpy', pad_mode="invalid")
        assert "pad_mode" in record.value.args[0]
