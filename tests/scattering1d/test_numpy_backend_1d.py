import pytest
import numpy as np
import scipy.fftpack

from kymatio.scattering1d.backend.numpy_backend import backend


def test_subsample_fourier():
    J = 10
    # 1d signal 
    x = np.random.randn(2, 2 ** J) + 1j * np.random.randn(2, 2 ** J)
    x_f = np.fft.fft(x, axis=-1)

    for j in range(J + 1):
        x_f_sub = backend.subsample_fourier(x_f, 2 ** j)
        x_sub = np.fft.ifft(x_f_sub, axis=-1)
        assert np.allclose(x[:, ::2 ** j], x_sub)

    with pytest.raises(TypeError) as te:
        x_bad = x.real
        backend.subsample_fourier(x_bad, 1)
    assert "should be complex" in te.value.args[0]


def test_pad():
    N = 128
    x = np.random.rand(2, 4, N)
    
    for pad_left in range(0, N - 16, 16):
        for pad_right in [pad_left, pad_left + 16]:
            x_pad = backend.pad(x, pad_left, pad_right)
            
            # compare left reflected part of padded array with left side 
            # of original array
            for t in range(1, pad_left + 1):
                assert np.allclose(x_pad[..., pad_left - t], x[..., t])
            # compare left part of padded array with left side of 
            # original array
            for t in range(x.shape[-1]):
                assert np.allclose(x_pad[..., pad_left + t], x[..., t])
            # compare right reflected part of padded array with right side
            # of original array
            for t in range(1, pad_right + 1):
                assert np.allclose(x_pad[..., x_pad.shape[-1] - 1 - pad_right + t], x[..., x.shape[-1] - 1 - t])
            # compare right part of padded array with right side of 
            # original array
            for t in range(1, pad_right + 1):
                assert np.allclose(x_pad[..., x_pad.shape[-1] - 1 - pad_right - t], x[..., x.shape[-1] - 1 - t])

    with pytest.raises(ValueError) as ve:
        backend.pad(x, x.shape[-1], 0)
    assert "padding size" in ve.value.args[0]
    
    with pytest.raises(ValueError) as ve:
        backend.pad(x, 0, x.shape[-1])
    assert "padding size" in ve.value.args[0]


def test_unpad():
    # test unpading of a random tensor
    x = np.random.rand(8, 4)
    
    y = backend.unpad(x, 1, 3)

    assert y.shape == (8, 2)
    assert np.allclose(y, x[:, 1:3])

    N = 128
    x = np.random.rand(2, 4, N)

    # similar to for loop in pad test
    for pad_left in range(0, N - 16, 16):
        pad_right = pad_left + 16
        x_pad = backend.pad(x, pad_left, pad_right)
        x_unpadded = backend.unpad(x_pad, pad_left, x_pad.shape[-1] - pad_right)
        assert np.allclose(x, x_unpadded)


def test_fft_type():
    x = np.random.rand(8, 4) + 1j * np.random.rand(8, 4)

    with pytest.raises(TypeError) as record:
        y = backend.rfft(x)
    assert 'should be real' in record.value.args[0]

    x = np.random.rand(8, 4) 

    with pytest.raises(TypeError) as record:
        y = backend.ifft(x)
    assert 'should be complex' in record.value.args[0]

    with pytest.raises(TypeError) as record:
        y = backend.irfft(x)
    assert 'should be complex' in record.value.args[0]


def test_fft():
    def coefficent(n):
            return np.exp(-2 * np.pi * 1j * n)

    x_r = np.random.rand(4)

    I, K = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')

    coefficents = coefficent(K * I / x_r.shape[0])
        
    y_r = (x_r * coefficents).sum(-1)

    z = backend.rfft(x_r)
    assert np.allclose(y_r, z)

    z_1 = backend.ifft(z)
    assert np.allclose(x_r, z_1)

    z_2 = backend.irfft(z)
    assert not np.iscomplexobj(z_2)
    assert np.allclose(x_r, z_2)
