"""
Testing all functions in filters_bank
"""

from kymatio.scattering1d.filter_bank import (adaptive_choice_P, anden_generator,
    compute_sigma_psi, compute_temporal_support, compute_xi_max, morlet_1d,
    get_max_dyadic_subsampling, gauss_1d, spin)
import numpy as np
import math
import pytest


def test_adaptive_choice_P():
    """
    Tests whether adaptive_choice_P provides a bound P which satisfies
    the adequate requirements
    """
    sigma_range = np.logspace(-5, 2, num=10)
    eps_range = np.logspace(-10, -5, num=8)
    for i in range(sigma_range.size):
        for j in range(eps_range.size):
            sigma = sigma_range[i]
            eps = eps_range[j]
            # choose the formula
            P = adaptive_choice_P(sigma, eps=eps)
            # check at the boundaries
            denom = 2 * (sigma**2)
            lim_left = np.exp(-((1 - P)**2) / denom)
            lim_right = np.exp(-(P**2) / denom)
            assert lim_left <= eps
            assert lim_right <= eps


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


def test_compute_xi_max():
    """
    Tests that 0.25 <= xi_max(Q) <= 0.5, whatever Q
    """
    Q_range = np.arange(1, 21, dtype=int)
    for Q in Q_range:
        xi_max = compute_xi_max(Q)
        assert xi_max <= 0.5
        assert xi_max >= 0.25


def test_get_max_dyadic_subsampling():
    """
    Tests on the subsampling formula for wavelets, to check that the retained
    value does not create aliasing (the wavelet should have decreased by
    a relative value of 1e-2 at the border of the aliasing.)
    """
    N = 2**12
    Q_range = np.arange(1, 20, dtype=int)
    J = 7
    for Q in Q_range:
        xi_max = compute_xi_max(Q)
        xi_range = xi_max * np.power(0.5, np.arange(J * Q) / float(Q))
        for xi in xi_range:
            sigma = compute_sigma_psi(xi, Q)
            j = get_max_dyadic_subsampling(xi, sigma, alpha=5.)
            # Check for subsampling. If there is no subsampling, the filters
            # cannot be aliased, so no need to check them.
            if j > 0:
                # compute the corresponding Morlet
                psi_f = morlet_1d(N, xi, sigma)
                # find the integer k such that
                k = N // 2**(j + 1)
                assert np.abs(psi_f[k]) / np.max(np.abs(psi_f)) < 1e-2


def test_compute_temporal_support():
    # Define constant averaging filter. This will be "too long" to avoid
    # border effects.
    h_f = np.fft.fft(np.ones((1, 4)), axis=1)
    with pytest.warns(UserWarning) as record:
        compute_temporal_support(h_f)
    assert "too small to avoid border effects" in record[0].message.args[0]


def test_spin():
    J = 5
    Q = 1
    filterbank_kwargs = {"alpha": 5, "r_psi": math.sqrt(0.5), "sigma0": 0.1}
    unspinned_xisigmas = list(anden_generator(J, Q, **filterbank_kwargs))
    spinned_generator, spinned_kwargs = spin(anden_generator, filterbank_kwargs)
    spinned_xisigmas = list(spinned_generator(J, Q, **spinned_kwargs))
    assert len(spinned_xisigmas) == (2*len(unspinned_xisigmas))
    assert spinned_xisigmas[0][0] == -spinned_xisigmas[J+1][0]
    assert spinned_xisigmas[0][1] == spinned_xisigmas[J+1][1]
    assert spinned_kwargs == filterbank_kwargs
