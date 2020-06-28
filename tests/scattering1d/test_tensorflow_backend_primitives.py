import pytest
import numpy as np

from kymatio.scattering1d.backend.tensorflow_backend import backend

def test_subsample_fourier(random_state=42):
    rng = np.random.RandomState(random_state)
    J = 10
    # 1d signal 
    x = rng.randn(2, 2**J) + 1j * rng.randn(2, 2**J)
    x_f = np.fft.fft(x, axis=-1)

    for j in range(J + 1):
        x_f_sub = backend.subsample_fourier(x_f, 2**j)
        x_sub = np.fft.ifft(x_f_sub, axis=-1)
        assert np.allclose(x[:, ::2**j], x_sub)

    with pytest.raises(TypeError) as te:
        x_bad = x.real
        backend.subsample_fourier(x_bad, 1)
    assert "should be complex" in te.value.args[0]


