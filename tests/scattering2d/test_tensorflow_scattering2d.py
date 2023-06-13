import os
import io
from kymatio import Scattering2D
import numpy as np
import pytest
from collections import namedtuple

backends = []

from kymatio.scattering2d.backend.tensorflow_backend import backend
backends.append(backend)

class TestScattering2DTensorFlow:
    @pytest.mark.parametrize('backend', backends)
    def test_Scattering2D(self, backend):
        test_data_dir = os.path.dirname(__file__)
        data = None
        with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)

        x = data['x']
        S = data['Sx']
        J = data['J']
        pre_pad = data['pre_pad']

        M = x.shape[2]
        N = x.shape[3]

        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad,
                                  frontend='tensorflow', backend=backend)

        x = x
        S = S
        Sg = scattering(x)
        assert np.allclose(Sg, S)

        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad,
                                  max_order=1, frontend='tensorflow',
                                  backend=backend)

        S1x = scattering(x)
        assert np.allclose(S1x, S[..., :S1x.shape[-3], :, :])

    @pytest.mark.parametrize('backend', backends)
    def test_batch_shape_agnostic(self, backend):
        J = 3
        L = 8
        shape = (32, 32)

        shape_ds = tuple(n // (2 ** J) for n in shape)

        S = Scattering2D(J, shape, L, backend=backend, frontend='tensorflow')

        x = np.zeros(shape)

        Sx = S(x)

        assert len(Sx.shape) == 3
        assert Sx.shape[-2:] == shape_ds

        n_coeffs = Sx.shape[-3]

        test_shapes = ((1,) + shape, (2,) + shape, (2, 2) + shape,
                       (2, 2, 2) + shape)

        for test_shape in test_shapes:
            x = np.zeros(test_shape)

            Sx = S(x)

            assert len(Sx.shape) == len(test_shape) + 1
            assert Sx.shape[-2:] == shape_ds
            assert Sx.shape[-3] == n_coeffs
            assert Sx.shape[:-3] == test_shape[:-2]

    @pytest.mark.parametrize('backend', backends)
    def test_scattering2d_errors(self, backend):
        S = Scattering2D(3, (32, 32), frontend='tensorflow', backend=backend)

        with pytest.raises(TypeError) as record:
            S(None)
        assert 'input should be' in record.value.args[0]

        x = np.random.randn(32)

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert 'have at least two dimensions' in record.value.args[0]

        x = np.random.randn(31, 31)

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert 'Tensor must be of spatial size' in record.value.args[0]

        S = Scattering2D(3, (32, 32), pre_pad=True, frontend='tensorflow',
                         backend=backend)

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert 'Padded tensor must be of spatial size' in record.value.args[0]


    def test_inputs(self):
        fake_backend = namedtuple('backend', ['name',])
        fake_backend.name = 'fake'

        with pytest.raises(ImportError) as ve:
            scattering = Scattering2D(2, shape=(10, 10), frontend='tensorflow', backend=fake_backend)
        assert 'not supported' in ve.value.args[0]

        with pytest.raises(RuntimeError) as ve:
            scattering = Scattering2D(10, shape=(10, 10), frontend='tensorflow')
        assert 'smallest dimension' in ve.value.args[0]

class TestScatteringPad2DTensorFlow:
    @pytest.mark.parametrize('backend', backends)
    def test_Scattering2D_pad(self, backend):
        scattering = Scattering2D(J=5, shape=(32, 32), pre_pad=False, frontend="tensorflow", backend=backend)
        test_im = np.ones((1,1,32,32))
        test = scattering(test_im)
