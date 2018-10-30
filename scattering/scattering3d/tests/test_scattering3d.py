""" This script will test the submodules used by the scattering module"""
import torch
import numpy as np
from scattering import Scattering3D
from scattering.scattering3d import utils as sl

def linfnorm(x,y):
    return torch.max(torch.abs(x-y))

def rerror(x,y):
    nx= np.sum(np.abs(x))
    if nx==0:
        return np.sum(np.abs(y))
    else:
        return np.sum(np.abs(x-y))/nx


def test_FFT3d_central_freq_batch():
    # Checked the 0 frequency for the 3D FFT
    for gpu in [ True, False]:

        x = torch.FloatTensor(1, 32, 32, 32, 2).fill_(0)
        if gpu:
            x = x.cuda()

        a = x.sum()
        fft3d = sl.Fft3d()
        y = fft3d(x)
        c = y[:,0,0,0].sum()
        assert rerror(a,c)<1e-6


def test_solid_harmonic_scattering():
    # Compare value to analytical formula in the case of a single Gaussian
    centers = torch.FloatTensor(1, 1, 3).fill_(0)
    weights = torch.FloatTensor(1, 1).fill_(1)
    sigma_gaussian = 3.
    sigma_0_wavelet = 3.
    M, N, O, J, L = 128, 128, 128, 1, 3
    grid = torch.from_numpy(
        np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
    x = sl.generate_weighted_sum_of_gaussians(grid, centers, weights, sigma_gaussian)
    scat = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma_0_wavelet)
    args = {'integral_powers': [1]}
    s = scat(x, order_2=False, method='integral', method_args=args)

    for j in range(J+1):
        sigma_wavelet = sigma_0_wavelet*2**j
        k = sigma_wavelet / np.sqrt(sigma_wavelet**2 + sigma_gaussian**2)
        for l in range(1, L+1):
            assert rerror(s[0, 0, j, l],k**l)<1e-4
