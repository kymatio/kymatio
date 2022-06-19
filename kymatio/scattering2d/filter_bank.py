import numpy as np
from scipy.fft import fft2, ifft2


def filter_bank(M, N, J, L=8):
    """
        Builds in Fourier the Morlet filters used for the scattering transform.
        Each single filter is provided as a dictionary with the following keys:
        * 'j' : scale
        * 'theta' : angle used
        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    filters = {}
    filters['psi'] = []

    for j in range(J):
        for theta in range(L):
            psi = {'levels': [], 'j': j, 'theta': theta}
            psi_signal = morlet_2d(M, N, 0.8 * 2**j,
                (int(L-L/2-1)-theta) * np.pi / L,
                3.0 / 4.0 * np.pi /2**j, 4.0/L)
            psi_signal_fourier = np.real(fft2(psi_signal))
            # drop the imaginary part, it is zero anyway
            psi_levels = []
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_levels.append(periodize_filter_fft(psi_signal_fourier, res))
            psi['levels'] = psi_levels
            filters['psi'].append(psi)

    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = np.real(fft2(phi_signal))
    # drop the imaginary part, it is zero anyway
    filters['phi'] = {'levels': [], 'j': J}
    for res in range(J):
        filters['phi']['levels'].append(
            periodize_filter_fft(phi_signal_fourier, res))

    return filters


def periodize_filter_fft(x, res):
    """
        Parameters
        ----------
        x : numpy array
            signal to periodize in Fourier
        res :
            resolution to which the signal is cropped.

        Returns
        -------
        crop : numpy array
            It returns a crop version of the filter, assuming that
             the convolutions will be done via compactly supported signals.
    """
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), x.dtype)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab /= norm_factor

    return gab


__all__ = ['filter_bank']
