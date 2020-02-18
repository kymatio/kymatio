"""
Testing all functions in filters_bank
"""
from kymatio.scattering1d.filter_bank import (adaptive_choice_P, periodize_filter_fourier, get_normalizing_factor,
    compute_sigma_psi, compute_temporal_support, compute_xi_max, morlet_1d, calibrate_scattering_filters,
    get_max_dyadic_subsampling, gauss_1d)
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


def test_normalizing_factor(random_state=42):
    """
    Tests whether the computation of the normalizing factor does the correct
    job (i.e. actually normalizes the signal in l1 or l2)
    """
    rng = np.random.RandomState(random_state)
    size_signal = [2**j for j in range(5, 13)]
    norm_type = ['l1', 'l2']
    for N in size_signal:
        x = rng.randn(N) + 1j * rng.randn(N)
        x_f = np.fft.fft(x)
        for norm in norm_type:
            kappa = get_normalizing_factor(x_f, norm)
            x_norm = kappa * x
            if norm == 'l1':
                assert np.isclose(np.sum(np.abs(x_norm)) - 1, 0.)
            elif norm == 'l2':
                assert np.isclose(np.sqrt(np.sum(np.abs(x_norm)**2)) - 1., 0.)

    with pytest.raises(ValueError) as ve:
        get_normalizing_factor(np.zeros(4))
    assert "Zero division error is very likely" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        get_normalizing_factor(np.ones(4), normalize='l0')
    assert "normalizations only include" in ve.value.args[0]


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
    P_range = [1, 5]
    for N in size_signal:
        for Q in Q_range:
            xi_max = compute_xi_max(Q)
            xi_range = xi_max / np.power(2, np.arange(7))
            for xi in xi_range:
                for P in P_range:
                    sigma = compute_sigma_psi(xi, Q)
                    # get the morlet for these parameters
                    psi_f = morlet_1d(N, xi, sigma, normalize='l2', P_max=P)
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

    Q = 1
    xi = compute_xi_max(Q)
    sigma = compute_sigma_psi(xi, Q)

    with pytest.raises(ValueError) as ve:
        morlet_1d(size_signal[0], xi, sigma, P_max=5.1)
    assert "should be an int" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        morlet_1d(size_signal[0], xi, sigma, P_max=-5)
    assert "should be non-negative" in ve.value.args[0]


def test_gauss_1d():
    """
    Tests for Gabor low-pass
    - Make sure that it has a fast decay in time
    - Make sure that it is symmetric, up to 1e-7 absolute precision
    """
    N = 2**13
    J = 7
    P_range = [1, 5]
    sigma0 = 0.1
    tol = 1e-7
    for j in range(1, J + 1):
        for P in P_range:
            sigma_low = sigma0 / math.pow(2, j)
            g_f = gauss_1d(N, sigma_low, P_max=P)
            # check the symmetry of g_f
            assert np.max(np.abs(g_f[1:N // 2] - g_f[N // 2 + 1:][::-1])) < tol
            # make sure that it has a fast decay in time
            phi = np.fft.ifft(g_f)
            assert np.min(phi) > - tol
            assert np.min(np.abs(phi)) / np.max(np.abs(phi)) < 1e-4

    Q = 1
    xi = compute_xi_max(Q)
    sigma = compute_sigma_psi(xi, Q)

    with pytest.raises(ValueError) as ve:
        gauss_1d(N, xi, sigma, P_max=5.1)
    assert "should be an int" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        gauss_1d(N, xi, sigma, P_max=-5)
    assert "should be non-negative" in ve.value.args[0]


def test_calibrate_scattering_filters():
    """
    Various tests on the central frequencies xi and spectral width sigma
    computed for the scattering filterbank
    - Checks that all widths are > 0
    - Check that sigma_low is smaller than all sigma2
    """
    J_range = np.arange(2, 11)
    Q_range = np.arange(1, 21, dtype=int)
    for J in J_range:
        for Q in Q_range:
            sigma_low, xi1, sigma1, j1, xi2, sigma2, j2 = \
                calibrate_scattering_filters( J, Q)
            # Check that all sigmas are > 0
            assert sigma_low > 0
            for sig in sigma1:
                assert sig > 0
            for sig in sigma2:
                assert sig > 0
            # check that sigma_low is smaller than all sigma2
            for sig in sigma1:
                assert sig >= sigma_low
            for sig in sigma2:
                assert sig >= sigma_low

    with pytest.raises(ValueError) as ve:
        calibrate_scattering_filters(J_range[0], 0.9)
    assert "should always be >= 1" in ve.value.args[0]


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
            j = get_max_dyadic_subsampling(xi, sigma)
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
