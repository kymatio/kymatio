"""
Authors: Louis Thiry, Georgios Exarchakis and Michael Eickenberg
All rights reserved, 2017.
"""

__all__ = ['solid_harmonic_filter_bank']

import numpy as np
from scipy.special import sph_harm, factorial
from .utils import get_3d_angles, double_factorial, sqrt


def solid_harmonic_filter_bank(M, N, O, J, L, sigma_0, fourier=True):
    """
        Computes a set of 3D Solid Harmonic Wavelets of scales j = [0, ..., J]
        and first orders l = [0, ..., L].

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        L : int
            maximal first order of the wavelets
        sigma_0 : float
            width parameter of mother solid harmonic wavelet
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        filters : list of ndarray
            the element number l of the list is a torch array array of size
            (J+1, 2l+1, M, N, O, 2) containing the (J+1)x(2l+1) wavelets of order l.
    """
    filters = []
    for l in range(L + 1):
        filters_l = np.zeros((J + 1, 2 * l + 1, M, N, O), dtype='complex64')
        for j in range(J+1):
            sigma = sigma_0 * 2 ** j
            filters_l[j,...] = solid_harmonic_3d(M, N, O, sigma, l, fourier=fourier)
        filters.append(filters_l)
    return filters


def gaussian_filter_bank(M, N, O, J, sigma_0, fourier=True):
    """
        Computes a set of 3D Gaussian filters of scales j = [0, ..., J].

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        sigma_0 : float
            width parameter of father Gaussian filter
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        gaussians : ndarray
            torch array array of size (J+1, M, N, O, 2) containing the (J+1)
            Gaussian filters.
    """
    gaussians = np.zeros((J + 1, M, N, O), dtype='complex64')
    for j in range(J + 1):
        sigma = sigma_0 * 2 ** j
        gaussians[j, ...] = gaussian_3d(M, N, O, sigma, fourier=fourier)
    return gaussians


def gaussian_3d(M, N, O, sigma, fourier=True):
    """
        Computes a 3D Gaussian filter.

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        sigma : float
            gaussian width parameter
        fourier : boolean
            if true, the Gaussian if computed in Fourier space
	    if false, the Gaussian if computed in signal space

        Returns
        -------
        gaussian : ndarray
            numpy array of size (M, N, O) and type float32 ifftshifted such
            that the origin is at the point [0, 0, 0]
    """
    grid = np.fft.ifftshift(
        np.mgrid[-M // 2:-M // 2 + M,
                 -N // 2:-N // 2 + N,
                 -O // 2:-O // 2 + O].astype('float32'),
        axes=(1,2,3))
    _sigma = sigma
    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        _sigma = 1. / sigma

    gaussian = np.exp(-0.5 * (grid ** 2).sum(0) / _sigma ** 2)
    if not fourier:
        gaussian /= (2 * np.pi) ** 1.5 * _sigma ** 3

    return gaussian


def solid_harmonic_3d(M, N, O, sigma, l, fourier=True):
    """
        Computes a set of 3D Solid Harmonic Wavelets.
	A solid harmonic wavelet has two integer orders l >= 0 and -l <= m <= l
	In spherical coordinates (r, theta, phi), a solid harmonic wavelet is
	the product of a polynomial Gaussian r^l exp(-0.5 r^2 / sigma^2)
	with a spherical harmonic function Y_{l,m} (theta, phi).

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        sigma : float
            width parameter of the solid harmonic wavelets
        l : int
            first integer order of the wavelets
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        solid_harm : ndarray, type complex64
            numpy array of size (2l+1, M, N, 0) and type complex64 containing
            the 2l+1 wavelets of order (l , m) with -l <= m <= l.
            It is ifftshifted such that the origin is at the point [., 0, 0, 0]
    """
    solid_harm = np.zeros((2*l+1, M, N, O), np.complex64)
    grid = np.fft.ifftshift(
        np.mgrid[-M // 2:-M // 2 + M,
                 -N // 2:-N // 2 + N,
                 -O // 2:-O // 2 + O].astype('float32'),
        axes=(1,2,3))
    _sigma = sigma

    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        _sigma = 1. / sigma

    r_square = (grid ** 2).sum(0)
    r_power_l = sqrt(r_square ** l)
    gaussian = np.exp(-0.5 * r_square / _sigma ** 2).astype('complex64')

    if l == 0:
        if fourier:
            return gaussian.reshape((1, M, N, O))
        return gaussian.reshape((1, M, N, O)) / (
                                          (2 * np.pi) ** 1.5 * _sigma ** 3)

    polynomial_gaussian = r_power_l * gaussian / _sigma ** l

    polar, azimuthal = get_3d_angles(grid)

    for i_m, m in enumerate(range(-l, l + 1)):
        solid_harm[i_m] = sph_harm(m, l, azimuthal, polar) * polynomial_gaussian

    if l % 2 == 0:
        norm_factor = 1. / (2 * np.pi * np.sqrt(l + 0.5) * 
                                            double_factorial(l + 1))
    else :
        norm_factor = 1. / (2 ** (0.5 * ( l + 3)) * 
                            np.sqrt(np.pi * (2 * l + 1)) * 
                            factorial((l + 1) / 2))

    if fourier:
        norm_factor *= (2 * np.pi) ** 1.5 * (-1j) ** l
    else:
        norm_factor /= _sigma ** 3

    solid_harm *= norm_factor

    return solid_harm
