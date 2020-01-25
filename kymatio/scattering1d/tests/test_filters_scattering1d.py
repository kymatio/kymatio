"""
Testing all functions in filters_bank
"""
from kymatio.scattering1d.filter_bank import (
    gauss_1d, morlet_1d, periodize_filter_fourier)
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
    xi_range = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    sigma_over_xi_range = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    for N in size_signal:
        for xi in sigma_range:
            for xi_over_sigma in xi_over_sigma_range:
                sigma = xi * sigma_over_xi
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
