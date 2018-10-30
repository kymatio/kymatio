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


def relative_difference(a, b):
    return np.linalg.norm((a - b) / np.maximum(np.abs(a), np.abs(b)))


def test_first_qm7_molecules_scattering_coefs():
    file_path = os.path.abspath(os.path.dirname(__file__))
    order_0_ref_cpu = np.load(os.path.join(file_path, 'data/order_0_ref_cpu.npy'))
    order_1_ref_cpu = np.load(os.path.join(file_path, 'data/order_1_ref_cpu.npy'))
    order_2_ref_cpu = np.load(os.path.join(file_path, 'data/order_2_ref_cpu.npy'))

    order_0_ref_gpu = np.load(os.path.join(file_path, 'data/order_0_ref_gpu.npy'))
    order_1_ref_gpu = np.load(os.path.join(file_path, 'data/order_1_ref_gpu.npy'))
    order_2_ref_gpu = np.load(os.path.join(file_path, 'data/order_2_ref_gpu.npy'))

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

    grid_gpu = grid.cuda()
    pos_gpu = pos.cuda()
    charges_gpu = charges.cuda()

    densities_cpu = generate_weighted_sum_of_gaussians(
            grid, pos, charges, sigma, cuda=False)
    order_0_cpu = compute_integrals(densities_cpu, integral_powers)
    order_1_cpu, order_2_cpu = scattering(densities_cpu, order_2=True,
                            method='integral', integral_powers=integral_powers)
    order_0_diff_cpu = relative_difference(order_0_ref_cpu, order_0_cpu.numpy())
    order_1_diff_cpu = relative_difference(order_1_ref_cpu, order_1_cpu.numpy())
    order_2_diff_cpu = relative_difference(order_2_ref_cpu, order_2_cpu.numpy())

    print('CPU', order_0_diff_cpu, order_1_diff_cpu, order_2_diff_cpu)
    assert  order_0_diff_cpu < 1e-3, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
    assert  order_1_diff_cpu < 1e-3, "CPU : order 1 do not match, diff={}".format(order_1_diff_cpu)
    assert  order_2_diff_cpu < 1e-3, "CPU : order 2 do not match, diff={}".format(order_2_diff_cpu)

    densities_gpu = generate_weighted_sum_of_gaussians(
            grid_gpu, pos_gpu, charges_gpu, sigma, cuda=True)
    order_0_gpu = compute_integrals(densities_gpu, integral_powers)
    order_1_gpu, order_2_gpu = scattering(densities_gpu, order_2=True,
                            method='integral', integral_powers=integral_powers)
    order_0_diff_gpu = relative_difference(order_0_ref_gpu, order_0_gpu.cpu().numpy())
    order_1_diff_gpu = relative_difference(order_1_ref_gpu, order_1_gpu.cpu().numpy())
    order_2_diff_gpu = relative_difference(order_2_ref_gpu, order_2_gpu.cpu().numpy())

    print('GPU', order_0_diff_gpu, order_1_diff_gpu, order_2_diff_gpu)
    assert  order_0_diff_gpu < 1e-4, "GPU : order 0 do not match, diff={}".format(order_0_diff_gpu)
    assert  order_1_diff_gpu < 1e-4, "GPU : order 1 do not match, diff={}".format(order_1_diff_gpu)
    assert  order_2_diff_gpu < 1e-4, "GPU : order 2 do not match, diff={}".format(order_2_diff_gpu)


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

