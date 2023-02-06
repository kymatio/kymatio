import pytest
from kymatio import Scattering1D
import io
import os
import numpy as np
import tensorflow as tf
from kymatio.scattering1d.backend.tensorflow_backend import backend

backends = [backend]

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
        S = Scattering1D(
            J, shape, Q=0.9, backend=backend, frontend='tensorflow')
        Q = S.Q
    assert "Q must always be >= 1" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(
            J, shape, Q=[8], backend=backend, frontend='tensorflow')
        Q = S.Q
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


@pytest.mark.parametrize("backend", backends)
def test_Scattering1D_average_global(backend):
    """
    Tests global averaging.
    """
    N = 2 ** 13
    Q = (1, 1)
    J = 5
    T = 'global'
    sc = Scattering1D(J, N, Q, T, backend=backend, frontend='tensorflow', out_type='array')
    x = tf.zeros((N,))
    Sx = sc(x).numpy()
    assert Sx.shape[-1] == 1


@pytest.mark.parametrize("backend", backends)
def test_differentiability_scattering(backend, random_state=42):
    """
    It simply tests whether it is really differentiable or not.
    This does NOT test whether the gradients are correct.
    """

    tf.random.set_seed(random_state) 

    J = 6
    Q = 8
    T = 2**12
    
    scattering = Scattering1D(J, T, Q, frontend='tensorflow', backend=backend)
    x = tf.Variable(tf.random.normal((2, T)))

    with tf.GradientTape(persistent=True) as tape:
        s = scattering(x)
        loss = tf.reduce_sum(tf.abs(s))
    grad = tape.gradient(loss, x)
    assert tf.reduce_max(tf.abs(grad)) > 0.
