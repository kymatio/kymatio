"""
Testing all functions in filters_bank
"""
from scattering.scattering1d.filter_bank import adaptative_choice_P
from scattering.scattering1d.filter_bank import periodize_filter_fft
from scattering.scattering1d.filter_bank import get_normalizing_factor
from scattering.scattering1d.filter_bank import compute_sigma_psi
from scattering.scattering1d.filter_bank import compute_xi_max
from scattering.scattering1d.filter_bank import morlet1D
from scattering.scattering1d.filter_bank import calibrate_scattering_filters
from scattering.scattering1d.filter_bank import get_max_dyadic_subsampling
from scattering.scattering1d.filter_bank import gauss1D
import numpy as np
from sklearn.utils import check_random_state
import math


def test_adaptative_choice_P():
    """
    Testing whether adaptive choice_P provides a bound P which satisfies
    the adequate requirements
    """
    sigma_range = np.logspace(-5, 2, num=10)
    eps_range = np.logspace(-10, -5, num=8)
    for i in range(sigma_range.size):
        for j in range(eps_range.size):
            sigma = sigma_range[i]
            eps = eps_range[j]
            # choose the formula
            P = adaptative_choice_P(sigma, eps=eps)
            # check at the boundaries
            denom = 2 * (sigma**2)
            lim_left = np.exp(-((1 - P)**2) / denom)
            lim_right = np.exp(-(P**2) / denom)
            assert lim_left <= eps
            assert lim_right <= eps


def test_periodize_filter_fft(random_state=42):
    """
    Tests whether the periodization in Fourier corresponds to
    a subsampling in time
    """
    rng = check_random_state(random_state)
    size_signal = [2**j for j in range(5, 10)]
    periods = [2**k for k in range(0, 6)]

    for N in size_signal:
        x = rng.randn(N) + 1j * rng.randn(N)
        x_fft = np.fft.fft(x)
        for per in periods:
            x_per_fft = periodize_filter_fft(x_fft, nperiods=per)
            x_per = np.fft.ifft(x_per_fft)
            assert np.max(np.abs(x_per - x[::per])) < 1e-7


def test_normalizing_factor(random_state=42):
    """
    Tests whether the computation of the normalizing factor does the correct
    job (i.e. actually normalizes the signal in l1 or l2)
    """
    rng = check_random_state(random_state)
    size_signal = [2**j for j in range(5, 10)]
    norm_type = ['l1', 'l2']
    for N in size_signal:
        x = rng.randn(N) + 1j * rng.randn(N)
        x_fft = np.fft.fft(x)
        for norm in norm_type:
            kappa = get_normalizing_factor(x_fft, norm)
            x_norm = kappa * x
            if norm == 'l1':
                assert np.isclose(np.sum(np.abs(x_norm)) - 1, 0.)
            elif norm == 'l2':
                assert np.isclose(np.sqrt(np.sum(np.abs(x_norm)**2)) - 1., 0.)


def test_morlet1D():
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
                psi_fft = morlet1D(N, xi, sigma, normalize='l2')
                # make sure that it has zero mean
                assert np.isclose(psi_fft[0], 0.)
                # make sure that it has a fast decay in time
                psi = np.fft.ifft(psi_fft)
                psi_abs = np.abs(psi)
                assert np.min(psi_abs) / np.max(psi_abs) < 1e-4
                # Check that the maximal frequency is relatively close to xi,
                # up to 1 percent
                k_max = np.argmax(np.abs(psi_fft))
                xi_emp = float(k_max) / float(N)
                assert np.abs(xi_emp - xi) / xi < 1e-2


def test_gauss1D():
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
        g_fft = gauss1D(N, sigma_low)
        # check the symmetry of g_fft
        assert np.max(np.abs(g_fft[1:N // 2] - g_fft[N // 2 + 1:][::-1])) < tol
        # make sure that it has a fast decay in time
        phi = np.fft.ifft(g_fft)
        assert np.min(phi) > - tol
        assert np.min(np.abs(phi)) / np.max(np.abs(phi)) < 1e-4


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
            sigma_low, xi1, sigma1, xi2, sigma2 = calibrate_scattering_filters(
                J, Q)
            # Check that all sigmas are > 0
            assert sigma_low > 0
            for sig in sigma1.values():
                assert sig > 0
            for sig in sigma2.values():
                assert sig > 0
            # check that sigma_low is smaller than all sigma2
            for sig in sigma1.values():
                assert sig >= sigma_low
            for sig in sigma2.values():
                assert sig >= sigma_low


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
            if j > 0:  # if there is subsampling
                # compute the corresponding Morlet
                psi_fft = morlet1D(N, xi, sigma)
                # find the integer k such that
                k = N // 2**(j + 1)
                assert np.abs(psi_fft[k]) / np.max(np.abs(psi_fft)) < 1e-2
            else:
                # pass this case: we have detected that there cannot
                # be any subsampling, and we assume that the filters are not
                # aliased already
                pass
