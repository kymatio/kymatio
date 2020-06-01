import pytest
from kymatio import Scattering1D
import os
import numpy
import jax.numpy as np
from jax import device_put
import io

backends = []

from kymatio.scattering1d.backend.jax_backend import backend
backends.append(backend)

class TestScattering1DJax:
    @pytest.mark.parametrize('backend', backends)
    def test_Scattering1D(self, backend):
        """
        Applies scattering on a stored signal to make sure its output agrees with
        a previously calculated version.
        """
        test_data_dir = os.path.dirname(__file__)

        with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
            buffer = io.BytesIO(f.read())
            data = numpy.load(buffer)

        x = device_put(np.asarray(data['x']))
        J = data['J']
        Q = data['Q']
        Sx0 = device_put(np.asarray(data['Sx']))

        T = x.shape[-1]
        print(x.dtype, J.dtype, Q.dtype, Sx0.dtype)
        scattering = Scattering1D(J, T, Q, backend=backend, frontend='jax')

        Sx = scattering(x)
        #print(Sx.dtype)
        #print(Sx-Sx0)
        assert np.allclose(Sx, Sx0)

#class TestScattering1DJaxSubsampleFourier:
#    def test_subsample_fourier(self):

