import pytest
import numpy as np
import scipy.signal
from kymatio.numpy import Scattering1D


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
    else:
        pytest.main([__file__, "-s"])
