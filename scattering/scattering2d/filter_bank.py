"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['filter_bank']

import torch
import numpy as np
import scipy.fftpack as fft




def filter_bank_real(M, N, J, L=8):
    filters = {}
    filters['psi'] = []


    offset_unpad = 0
    for j in range(J):
        for theta in range(L):
            psi = {}
            psi['j'] = j
            psi['theta'] = theta
            psi_signal = morlet_2d(M, N, 0.8 * 2**j, (int(L-L/2-1)-theta) * np.pi / L, 3.0 / 4.0 * np.pi /2**j, 4.0/L, offset=offset_unpad)
            psi_signal_fourier = fft.fft2(psi_signal)
            for res in range(j + 1):
                psi_signal_fourier_res = periodize_filter_fft(psi_signal_fourier, res)
                psi[res]=torch.FloatTensor(np.stack((np.real(psi_signal_fourier_res), np.imag(psi_signal_fourier_res)), axis=2))
                # Normalization to avoid doing it with the FFT!
                psi[res].div_(M*N// 2**(2*j))
            filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0, offset=offset_unpad)
    phi_signal_fourier = fft.fft2(phi_signal)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = periodize_filter_fft(phi_signal_fourier, res)
        filters['phi'][res]=torch.FloatTensor(np.stack((np.real(phi_signal_fourier_res), np.imag(phi_signal_fourier_res)), axis=2))
        filters['phi'][res].div_(M*N // 2 ** (2 * J))

    return filters


def filter_bank(M, N, J, L=8, cache=False):
    '''
    Cache filters to a file

    cache: string or False
        path to filter bank. 
        If parameters (M, N, J, L) match, load from cache, otherwise, recompute and overwrite.
    '''
    if not cache:
        return filter_bank_real(M, N, J, L)
    try:
        print('Attempting to load from ',cache,' ...')
        data = torch.load(cache)
        assert M == data['M'], 'M mismatch'
        assert N == data['N'], 'N mismatch'
        assert J == data['J'], 'J mismatch'
        assert L == data['L'], 'L mismatch'
        filters = data['filters']
        print('Loaded.')
        return filters
    except Exception as e:
        print('Load Error: ',e)
        print('(Re-)computing filters.')
        filters = filter_bank_real(M, N, J, L)
        print('Attempting to save to ',cache,' ...')
        try:
            with open(cache, 'wb') as fp:
                data = {'M':M, 'N':N, 'J':J, 'L':L, 'filters':filters}
            torch.save(data, cache)
            print('Saved.')
        except Exception as f:
            print('Save Error: ',f)
        return filters


def periodize_filter_fft(x, res):
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), np.complex64)

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


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=None):
    """ This function generated a morlet"""
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
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
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab
