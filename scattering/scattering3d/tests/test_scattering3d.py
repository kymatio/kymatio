""" This script will test the submodules used by the scattering module"""
import torch
import numpy as np
import os.path
from scattering import Scattering3D
from scattering.datasets import get_qm7_positions_and_charges
from scattering.scattering3d.utils import generate_weighted_sum_of_gaussians, compute_integrals

def linfnorm(x,y):
    return torch.max(torch.abs(x-y))

def rerror(x,y):
    nx= np.sum(np.abs(x))
    if nx==0:
        return np.sum(np.abs(y))
    else:
        return np.sum(np.abs(x-y))/nx


def test_first_qm7_molecules_scattering_coefs():
    file_path = os.path.abspath(os.path.dirname(__file__))
    order_0_ref = np.load(os.path.join(file_path, 'data/order_0_ref.npy'))
    order_1_ref = np.load(os.path.join(file_path, 'data/order_1_ref.npy'))
    order_2_ref = np.load(os.path.join(file_path, 'data/order_2_ref.npy'))
    M, N, O, J, L = 192, 128, 96, 2, 2
    integral_powers = [1., 2.]
    sigma = 1.5
    pos, full_charges, _ = get_qm7_positions_and_charges(sigma)
    pos = pos[:8]
    charges = full_charges[:8]

    scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

    grid = torch.from_numpy(
        np.fft.ifftshift(
            np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'),
            axes=(1, 2, 3)))

    for cuda in [True, False]:
        if cuda:
            _grid = grid.cuda()
            _pos = pos.cuda()
            _charges = charges.cuda()
        else:
            _grid = grid
            _pos = pos
            _charges = charges
        first_densities = generate_weighted_sum_of_gaussians(
                _grid, _pos, _charges, sigma, cuda=cuda)

        order_0 = compute_integrals(first_densities, integral_powers)
        order_1, order_2 = scattering(first_densities, order_2=True,
                                method='integral', integral_powers=integral_powers)

        import pdb;pdb.set_trace()
        assert rerror(order_0_ref, order_0.cpu().numpy()) < 1e-6
        assert rerror(order_1_ref, order_1.cpu().numpy()) < 1e-6
        assert rerror(order_2_ref, order_2.cpu().numpy()) < 1e-6


def test_solid_harmonic_scattering():
    # Compare value to analytical formula in the case of a single Gaussian
    centers = torch.FloatTensor(1, 1, 3).fill_(0)
    weights = torch.FloatTensor(1, 1).fill_(1)
    sigma_gaussian = 3.
    sigma_0_wavelet = 3.
    M, N, O, J, L = 128, 128, 128, 1, 3
    grid = torch.from_numpy(
        np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
    x = generate_weighted_sum_of_gaussians(grid, centers, weights, sigma_gaussian)
    scat = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma_0_wavelet)
    args = {'integral_powers': [1]}
    s = scat(x, order_2=False, method='integral', method_args=args)

    for j in range(J+1):
        sigma_wavelet = sigma_0_wavelet*2**j
        k = sigma_wavelet / np.sqrt(sigma_wavelet**2 + sigma_gaussian**2)
        for l in range(1, L+1):
            assert rerror(s[0, 0, j, l],k**l)<1e-4

