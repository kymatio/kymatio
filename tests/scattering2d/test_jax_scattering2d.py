import os
import io
import jax.numpy as jnp
import numpy as np
from kymatio import Scattering2D
from collections import namedtuple
import pytest


from kymatio.scattering2d.backend.jax_backend import backend


def test_Pad():
    pad = backend.Pad((2, 2, 2, 2), (4, 4))

    x = np.random.randn(1, 4, 4) + 1J * np.random.randn(1, 4, 4)

    z = pad(x)

    assert z.shape == (1, 8, 8)
    assert z[0, 2, 2] == x[0, 0, 0]
    assert z[0, 1, 0] == x[0, 1, 2]
    assert z[0, 1, 1] == x[0, 1, 1]
    assert z[0, 1, 2] == x[0, 1, 0]
    assert z[0, 1, 3] == x[0, 1, 1]


def test_unpad():
    x = np.random.randn(4, 4) + 1J * np.random.randn(4, 4)

    y = backend.unpad(x)

    assert y.shape == (2, 2)
    assert y[0, 0] == x[1, 1]
    assert y[0, 1] == x[1, 2]


def test_Modulus():
    modulus = backend.modulus

    x = np.random.rand(100, 10, 4) + 1J * np.random.rand(100, 10, 4)

    v = modulus(x)
    u = np.sqrt(x.real ** 2 + x.imag ** 2)
    assert np.allclose(u, v)


def test_SubsampleFourier():
    subsample_fourier = backend.subsample_fourier

    x = (np.random.rand(100, 128, 128)
            + 1J * np.random.rand(100, 128, 128))

    y = np.zeros((100, 8, 8), dtype=np.complex128)

    from itertools import product
    for i, j in product(range(8), range(8)):
        for m, n in product(range(16), range(16)):
            y[..., i, j] += x[..., i + m * 8, j + n * 8]

    y /= 16 ** 2

    z = subsample_fourier(x, k=16)
    assert np.allclose(y, z)


class TestCDGMM:
    @pytest.fixture(params=(False, True))
    def data(self, request):
        real_filter = request.param
        x = (np.random.randn(100, 128, 128)
             + 1J * np.random.randn(100, 128, 128))
        filt = (np.random.randn(128, 128)
                + 1J * np.random.randn(128, 128))
        y = (np.random.randn(100, 128, 128)
             + 1J * np.random.randn(100, 128, 128))

        if real_filter:
            filt = np.real(filt)

        y = x * filt

        return x, filt, y

    def test_cdgmm_forward(self, data):
        x, filt, y = data

        z = backend.cdgmm(x, filt)

        assert np.allclose(y, z)

def test_cdgmm_exceptions():
    with pytest.raises(TypeError) as record:
        backend.cdgmm(np.empty((3, 4, 5)).astype(np.float64),
                        np.empty((4, 5)).astype(np.complex128))
    assert 'first input must be complex' in record.value.args[0]

    with pytest.raises(TypeError) as record:
        backend.cdgmm(np.empty((3, 4, 5)).astype(np.complex128),
                        np.empty((4, 5)).astype(np.int64))
    assert 'second input must be complex or real' in record.value.args[0]

    with pytest.raises(RuntimeError) as record:
        backend.cdgmm(np.empty((3, 4, 5)).astype(np.complex128),
                        np.empty((4, 6)).astype(np.complex128))
    assert 'not compatible for multiplication' in record.value.args[0]


def test_fft():
    np.random.seed(9161341)
    x = np.random.randn(2, 2) + 1J * np.random.randn(2, 2)

    y = np.array([[x[0, 0] + x[0, 1] + x[1, 0] + x[1, 1],
                    x[0, 0] - x[0, 1] + x[1, 0] - x[1, 1]],
                    [x[0, 0] + x[0, 1] - x[1, 0] - x[1, 1],
                    x[0, 0] - x[0, 1] - x[1, 0] + x[1, 1]]])


    z = backend.ifft(x)

    z = z * 4

    assert np.allclose(y, z)

    z = backend.irfft(x)

    z = z * 4

    assert not np.iscomplexobj(z)
    assert np.allclose(y.real, z)

    x = np.random.randn(2, 2)

    y = np.array([[x[0, 0] + x[0, 1] + x[1, 0] + x[1, 1],
                    x[0, 0] - x[0, 1] + x[1, 0] - x[1, 1]],
                    [x[0, 0] + x[0, 1] - x[1, 0] - x[1, 1],
                    x[0, 0] - x[0, 1] - x[1, 0] + x[1, 1]]])

    z = backend.rfft(x)
    assert np.allclose(y.real, z)


def test_concatenate():
    x = np.random.randn(3, 6, 6) + 1J * np.random.randn(3, 6, 6)
    y = np.random.randn(3, 6, 6) + 1J * np.random.randn(3, 6, 6)
    z = np.random.randn(3, 6, 6) + 1J * np.random.randn(3, 6, 6)

    w = backend.concatenate((x, y, z))

    assert w.shape == (x.shape[0],) + (3,) + (x.shape[-2:])
    assert np.allclose(w[:, 0, ...], x)
    assert np.allclose(w[:, 1, ...], y)
    assert np.allclose(w[:, 2, ...], z)


def test_Scattering2D():
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

def test_batch_shape_agnostic():
    J = 3
    L = 8
    shape = (32, 32)

    shape_ds = tuple(n // (2 ** J) for n in shape)

    S = Scattering2D(J, shape, L, backend=backend, frontend='jax')

    x = jnp.zeros(shape)

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

def test_scattering2d_errors():
    S = Scattering2D(3, (32, 32), frontend='jax', backend=backend)

    with pytest.raises(TypeError) as record:
        S(None)
    assert 'input should be' in record.value.args[0]

    x = np.random.randn(32)
    x = jnp.asarray(x)

    with pytest.raises(RuntimeError) as record:
        S(x)
    assert 'have at least two dimensions' in record.value.args[0]

    x = np.random.randn(31, 31)
    x = jnp.asarray(x)

    with pytest.raises(RuntimeError) as record:
        S(x)
    assert 'NumPy array must be of spatial size' in record.value.args[0]

    S = Scattering2D(3, (32, 32), pre_pad=True, frontend='jax',
                        backend=backend)

    with pytest.raises(RuntimeError) as record:
        S(x)
    assert 'Padded array must be of spatial size' in record.value.args[0]


def test_inputs():
    fake_backend = namedtuple('backend', ['name',])
    fake_backend.name = 'fake'

    with pytest.raises(ImportError) as ve:
        scattering = Scattering2D(2, shape=(10, 10), frontend='jax', backend=fake_backend)
    assert 'not supported' in ve.value.args[0]

    with pytest.raises(RuntimeError) as ve:
        scattering = Scattering2D(10, shape=(10, 10), frontend='jax')
    assert 'smallest dimension' in ve.value.args[0]
