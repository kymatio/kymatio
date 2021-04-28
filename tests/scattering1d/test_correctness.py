import numpy as np
from kymatio.numpy import Scattering1D


def test_energy_dirac():
    """Test that zeroth + first-order coeffs' L1 norm sums to 1 for Dirac input.
    """
    N = 2048
    J = 8
    Q = 1
    x = np.zeros(N)
    x[N//2] = 1

    ts = Scattering1D(shape=N, J=J, Q=Q, average=True)
    ts_x = ts(x)

    # compute L1 norm
    ts_x_L1 = np.abs(ts_x).sum(axis=1)
    ts_x_L1 *= 2**J  # account for subsampling

    n_zeroth = 1
    n_first  = J + 1
    n_total  = n_zeroth + n_first
    th = 1e-1  # can do better with greater N
    max_adiff = np.max(np.abs(ts_x_L1[:n_total] - 1))
    assert max_adiff < th, "max_adiff > th ({} > {})".format(max_adiff, th)
