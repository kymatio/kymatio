import pytest
from kymatio import Scattering1D
import os
import numpy
import jax.numpy as np
from jax import device_put
import io

backends = []

from kymatio.scattering1d.backend.jax_backend import backend


def test_Scattering1D():
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
    scattering = Scattering1D(J, T, (Q, 1), backend=backend, frontend='jax')


    Sx = scattering(x) 

    assert np.allclose(Sx, Sx0, atol=1e-6, rtol=1e-7)

    assert scattering.backend.name == 'jax'
    