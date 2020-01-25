"""
Testing all functions in filters_bank
"""
from kymatio.scattering1d.filter_bank import (periodize_filter_fourier,
    compute_sigma_psi, compute_temporal_support, compute_xi_max, morlet_1d, calibrate_scattering_filters,
    get_max_dyadic_subsampling, gauss_1d)
import numpy as np
import math
import pytest


def test_periodize_filter_fourier(random_state=42):
    """
    Tests whether the periodization in Fourier corresponds to
    a subsampling in time
    """
    rng = np.random.RandomState(random_state)
    size_signal = [2**j for j in range(5, 10)]
    periods = [2**k for k in range(0, 6)]

    for N in size_signal:
        x = rng.randn(N) + 1j * rng.randn(N)
        x_f = np.fft.fft(x)
        for per in periods:
            x_per_f = periodize_filter_fourier(x_f, nperiods=per)
            x_per = np.fft.ifft(x_per_f)
            assert np.max(np.abs(x_per - x[::per])) < 1e-7


def test_morlet_1d():
    """
    Tests for Morlet wavelets:
    - Make sure that it has exact zero mean
    - Make sure that it has a fast decay in time
    - Check that the maximal frequency is relatively close to xi,
        up to 1% accuracy
    """
    size_signal = [2**13]
    Q_range = np.arange(1, 20, dtype=int)
    for N in size_signal:
        for Q in Q_range:
            xi_max = compute_xi_max(Q)
            xi_range = xi_max / np.power(2, np.arange(7))
            for xi in xi_range:
                sigma = compute_sigma_psi(xi, Q)
                # get the morlet for these parameters
                psi_f = morlet_1d(N, xi, sigma)
                # make sure that it has zero mean
                assert np.isclose(psi_f[0], 0.)
                # make sure that it has a fast decay in time
                psi = np.fft.ifft(psi_f)
                psi_abs = np.abs(psi)
                assert np.min(psi_abs) / np.max(psi_abs) < 1e-3
                # Check that the maximal frequency is relatively close to xi,
                # up to 1 percent
                k_max = np.argmax(np.abs(psi_f))
                xi_emp = float(k_max) / float(N)
                assert np.abs(xi_emp - xi) / xi < 1e-2


def test_gauss_1d():
    """
    Tests for Gabor low-pass
    - Make sure that it has a fast decay in time
    - Make sure that it is symmetric, up to 1e-7 absolute precision
    """
    N = 2**13
    J = 7
    sigma0 = 0.1
    tol = 1e-7
    for j in range(1, J + 1):
        sigma_low = sigma0 / math.pow(2, j)
        g_f = gauss_1d(N, sigma_low)
        # check the symmetry of g_f
        assert np.max(np.abs(g_f[1:N // 2] - g_f[N // 2 + 1:][::-1])) < tol
        # make sure that it has a fast decay in time
        phi = np.fft.ifft(g_f)
        assert np.min(phi) > - tol
        assert np.min(np.abs(phi)) / np.max(np.abs(phi)) < 1e-4


def test_compute_temporal_support():
    # Define constant averaging filter. This will be "too long" to avoid
    # border effects.
    h_f = np.fft.fft(np.ones((1, 4)), axis=1)
    with pytest.warns(UserWarning) as record:
        compute_temporal_support(h_f)
    assert "too small to avoid border effects" in record[0].message.args[0]
