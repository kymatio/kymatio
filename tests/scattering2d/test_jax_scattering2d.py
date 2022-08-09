import os
import io
import jax.numpy as np
from kymatio import Scattering2D
import pytest
from jax import device_put


from kymatio.scattering2d.backend.jax_backend import backend


def test_Scattering2D():
    test_data_dir = os.path.dirname(__file__)
    data = None
    with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)

    x = device_put(np.array(data['x']))
    S = device_put(np.array(data['Sx']))
    J = data['J']
    pre_pad = data['pre_pad']

    M = x.shape[2]
    N = x.shape[3]

    scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad,
                                frontend='jax', backend=backend)

    x = x
    S = S
    Sg = scattering(x)
    assert np.allclose(Sg, S)

    scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad,
                                max_order=1, frontend='jax',
                                backend=backend)

    S1x = scattering(x)
    assert np.allclose(S1x, S[..., :S1x.shape[-3], :, :])
