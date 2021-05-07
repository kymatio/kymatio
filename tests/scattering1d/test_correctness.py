import pytest
import numpy as np
import scipy.signal
import warnings
from kymatio.numpy import Scattering1D
from kymatio.scattering1d.backend.agnostic_backend import pad

# set True to execute all test functions without pytest
run_without_pytest = 0


def test_T():
    """Test that `T` controls degree of invariance as intended."""
    # configure scattering & signal
    J = 11
    Q = 16
    N = 4096
    width = N//8
    shift = N//8
    T0, T1 = N//2, N//4
    freq_fracs = (4, 8, 16, 32)

    # make signal & shifted
    window = scipy.signal.tukey(width, alpha=0.5)
    window = np.pad(window, (N - width) // 2)
    t  = np.linspace(0, 1, N, endpoint=False)
    x  = np.sum([np.cos(2*np.pi * N/ff * t) for ff in freq_fracs], axis=0
                ) * window
    xs = np.roll(x, shift)

    # make scattering objects
    kw = dict(J=J, Q=Q, shape=N, average=1, out_type="array", pad_mode="zero",
              max_pad_factor=1)
    ts0 = Scattering1D(T=T0, **kw)
    ts1 = Scattering1D(T=T1, **kw)

    # scatter
    ts0_x  = ts0.scattering(x)
    ts0_xs = ts0.scattering(xs)
    ts1_x  = ts1.scattering(x)
    ts1_xs = ts1.scattering(xs)

    # compare distances
    l2_00_xxs = l2(ts0_x, ts0_xs)
    l2_11_xxs = l2(ts1_x, ts1_xs)

    th0, th1 = .021, .15
    assert l2_00_xxs < th0, "{} > {}".format(l2_00_xxs, th0)
    assert l2_11_xxs > th1, "{} < {}".format(l2_11_xxs, th1)


def _test_padding(backend_name):
    def _arange(N):
        if backend_name == 'tensorflow':
            return backend.range(N)
        return backend.arange(N)

    if backend_name == 'numpy':
        backend = np
    elif backend_name == 'torch':
        import torch
        backend = torch
    elif backend_name == 'tensorflow':
        import tensorflow as tf
        backend = tf

    for N in (128, 129):  # even, odd
        x = backend.reshape(_arange(6 * N), (2, 3, N))
        for pad_factor in (1, 2, 3, 4):
            pad_left = (N // 2) * pad_factor
            pad_right = int(np.ceil(N / 4) * pad_factor)

            for pad_mode in ('zero', 'reflect'):
                out0 = pad(x, pad_left, pad_right, pad_mode=pad_mode,
                           backend_name=backend_name)
                out1 = np.pad(x,
                              [[0, 0]] * (x.ndim - 1) + [[pad_left, pad_right]],
                              mode=pad_mode if pad_mode != 'zero' else 'constant')

                out0 = out0.numpy() if hasattr(out0, 'numpy') else out0
                assert np.allclose(out0, out1), (
                    "{} | (N, pad_mode, pad_left, pad_right) = ({}, {}, {}, {})"
                    ).format(backend_name, N, pad_mode, pad_left, pad_right)


def test_pad_numpy():
    _test_padding('numpy')


def test_pad_torch():
    try:
        import torch
    except ImportError:
        warnings.warn("Failed to import torch")
        return
    _test_padding('torch')


def test_pad_tensorflow():
    try:
        import tensorflow
    except ImportError:
        warnings.warn("Failed to import tensorflow")
        return
    _test_padding('tensorflow')


def _l2(x):
    return np.sqrt(np.sum(np.abs(x)**2))

def l2(x0, x1):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    return _l2(x1 - x0) / _l2(x0)


if __name__ == '__main__':
    if run_without_pytest:
        test_T()
        test_pad_numpy()
        test_pad_torch()
        test_pad_tensorflow()
    else:
        pytest.main([__file__, "-s"])
