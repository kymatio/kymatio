import numpy as np
import pytest

backends = []

from kymatio.scattering2d.backend.numpy_backend import backend
backends.append(backend)


class TestPad:
    @pytest.mark.parametrize('backend', backends)
    def test_Pad(self, backend):
        pad = backend.Pad((2, 2, 2, 2), (4, 4))

        x = np.random.randn(1, 4, 4) + 1J * np.random.randn(1, 4, 4)

        z = pad(x)

        assert z.shape == (1, 8, 8)
        assert z[0, 2, 2] == x[0, 0, 0]
        assert z[0, 1, 0] == x[0, 1, 2]
        assert z[0, 1, 1] == x[0, 1, 1]
        assert z[0, 1, 2] == x[0, 1, 0]
        assert z[0, 1, 3] == x[0, 1, 1]

    @pytest.mark.parametrize('backend', backends)
    def test_unpad(self, backend):
        x = np.random.randn(4, 4) + 1J * np.random.randn(4, 4)

        y = backend.unpad(x)

        assert y.shape == (2, 2)
        assert y[0, 0] == x[1, 1]
        assert y[0, 1] == x[1, 2]


class TestModulus:
    @pytest.mark.parametrize('backend', backends)
    def test_Modulus(self, backend):
        modulus = backend.modulus

        x = np.random.rand(100, 10, 4) + 1J * np.random.rand(100, 10, 4)

        y = modulus(x)
        u = np.sqrt(np.real(x) ** 2 + np.imag(x) ** 2)
        v = y
        assert np.allclose(u, v)


class TestSubsampleFourier:
    @pytest.mark.parametrize('backend', backends)
    def test_SubsampleFourier(self, backend):
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
    
    @pytest.mark.parametrize('backend', backends)
    def test_SubsampleFourier_type(self, backend):
        with pytest.raises(TypeError) as te:
            x_bad = np.random.rand(100, 128, 128)
            backend.subsample_fourier(x_bad, 1)
        assert "should be complex" in te.value.args[0]


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

    @pytest.mark.parametrize('backend', backends)
    def test_cdgmm_forward(self, data, backend):
        x, filt, y = data

        z = backend.cdgmm(x, filt)

        assert np.allclose(y, z)

    @pytest.mark.parametrize('backend', backends)
    def test_cdgmm_exceptions(self, backend):
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


class TestFFT:
    @pytest.mark.parametrize('backend', backends)
    def test_fft(self, backend):
        x = np.random.randn(2, 2)

        y = np.array([[x[0, 0] + x[0, 1] + x[1, 0] + x[1, 1],
                       x[0, 0] - x[0, 1] + x[1, 0] - x[1, 1]],
                      [x[0, 0] + x[0, 1] - x[1, 0] - x[1, 1],
                       x[0, 0] - x[0, 1] - x[1, 0] + x[1, 1]]])

        z = backend.rfft(x)
        assert np.allclose(y, z)

        z_1 = backend.ifft(z)
        assert np.allclose(x, z_1)

        z_2 = backend.irfft(z)
        assert not np.iscomplexobj(z_2)
        assert np.allclose(x, z_2)
        
    @pytest.mark.parametrize('backend', backends)
    def test_fft_type(self, backend):
        x = np.random.rand(8, 4, 4) + 1J * np.random.rand(8, 4, 4)

        with pytest.raises(TypeError) as record:
            y = backend.rfft(x)
        assert 'should be real' in record.value.args[0]

        x = np.random.rand(8, 4, 4)

        with pytest.raises(TypeError) as record:
            y = backend.ifft(x)
        assert 'should be complex' in record.value.args[0]

        with pytest.raises(TypeError) as record:
            y = backend.irfft(x)
        assert 'should be complex' in record.value.args[0]

       
class TestBackendUtils:
    @pytest.mark.parametrize('backend', backends)
    def test_stack(self, backend):
        x = np.random.randn(3, 6, 6) + 1J * np.random.randn(3, 6, 6)
        y = np.random.randn(3, 6, 6) + 1J * np.random.randn(3, 6, 6)
        z = np.random.randn(3, 6, 6) + 1J * np.random.randn(3, 6, 6)

        w = backend.stack((x, y, z))

        assert w.shape == (x.shape[0],) + (3,) + (x.shape[-2:])
        assert np.allclose(w[:, 0, ...], x)
        assert np.allclose(w[:, 1, ...], y)
        assert np.allclose(w[:, 2, ...], z)
