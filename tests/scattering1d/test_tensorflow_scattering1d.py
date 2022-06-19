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
        J = int(data['J'])
        Q = int(data['Q'])
        Sx0 = data['Sx']

        T = x.shape[-1]

        scattering = Scattering1D(J, T, Q, backend=backend, frontend='tensorflow')

        Sx = scattering(x)
        assert np.allclose(Sx, Sx0, atol=1e-6, rtol =1e-7)

@pytest.mark.parametrize("backend", backends)
def test_Q(backend):
    J = 3
    length = 1024
    shape = (length,)

    # test different cases for Q
    with pytest.raises(ValueError) as ve:
        _ = Scattering1D(
            J, shape, Q=0.9, backend=backend, frontend='tensorflow')
    assert "Q should always be >= 1" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        _ = Scattering1D(
            J, shape, Q=[8], backend=backend, frontend='tensorflow')
    assert "Q must be an integer or a tuple" in ve.value.args[0]

    Sc_int = Scattering1D(J, shape, Q=(8, ), backend=backend, frontend='tensorflow')
    Sc_tuple = Scattering1D(J, shape, Q=(8, 1), backend=backend, frontend='tensorflow')

    assert Sc_int.Q == Sc_tuple.Q

    # test dummy input
    x = np.zeros(length)
    Sc_int_out = Sc_int.scattering(x)
    Sc_tuple_out = Sc_tuple.scattering(x)

    assert np.allclose(Sc_int_out, Sc_tuple_out)
    assert Sc_int_out.shape == Sc_tuple_out.shape