import numpy as np
import scipy.signal
from kymatio.numpy import Scattering1D


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
    ts0 = Scattering1D(J=J, Q=Q, shape=N, T=T0, out_type='array')
    ts1 = Scattering1D(J=J, Q=Q, shape=N, T=T1, out_type='array')

    # scatter
    ts0_x  = ts0.scattering(x)
    ts0_xs = ts0.scattering(xs)
    ts1_x  = ts1.scattering(x)
    ts1_xs = ts1.scattering(xs)

    # compare distances
    l1l2_00_xxs = l1l2(ts0_x, ts0_xs)
    l1l2_11_xxs = l1l2(ts1_x, ts1_xs)

    th0 = .13
    th1 = th0 * 2
    assert l1l2_00_xxs < th0, "{} > {}".format(l1l2_00_xxs, th0)
    assert l1l2_11_xxs > th1, "{} < {}".format(l1l2_11_xxs, th1)


def _l1l2(x):
    return np.sum(np.sqrt(np.mean(x**2, axis=1)), axis=0)

def l1l2(x0, x1):
    """Coeff distance measure; Thm 2.12 in https://arxiv.org/abs/1101.2286"""
    return _l1l2(x1 - x0) / _l1l2(x0)
