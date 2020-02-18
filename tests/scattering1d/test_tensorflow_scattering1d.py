import pytest
from kymatio import Scattering1D
import os
import numpy as np
import io


backends = []

from kymatio.scattering1d.backend.tensorflow_backend import backend
backends.append(backend)


class TestScattering1DTensorFlow:
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

        scattering = Scattering1D(J, T, Q, backend=backend, frontend='tensorflow')

        Sx = scattering(x)
        assert np.allclose(Sx, Sx0, atol=1e-6, rtol =1e-7)
