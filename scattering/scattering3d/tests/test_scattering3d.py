""" This script will test the submodules used by the scattering module"""
import torch
import numpy as np
import os.path
from scattering import Scattering3D
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
    return np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), np.sum(np.abs(b)))


def test_against_standard_computations():
    file_path = os.path.abspath(os.path.dirname(__file__))
    x = np.load(os.path.join(file_path, 'data/x.npy'))

    order_0_ref_cpu = np.load(os.path.join(file_path, 'data/order_0_ref_cpu.npy'))
    order_1_ref_cpu = np.load(os.path.join(file_path, 'data/order_1_ref_cpu.npy'))
    order_2_ref_cpu = np.load(os.path.join(file_path, 'data/order_2_ref_cpu.npy'))

    order_0_ref_gpu = np.load(os.path.join(file_path, 'data/order_0_ref_gpu.npy'))
    order_1_ref_gpu = np.load(os.path.join(file_path, 'data/order_1_ref_gpu.npy'))
    order_2_ref_gpu = np.load(os.path.join(file_path, 'data/order_2_ref_gpu.npy'))

    M, N, O, J, L, sigma = 64, 64, 64, 2, 2, 1.
    integral_powers = [1., 2.]

    scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

    x_cpu = torch.from_numpy(x)

    order_0_cpu = compute_integrals(x_cpu, integral_powers)
    order_1_cpu, order_2_cpu = scattering(x_cpu, order_2=True,
                            method='integral', integral_powers=integral_powers)

    order_0_diff_cpu = relative_difference(order_0_ref_cpu, order_0_cpu.numpy())
    order_1_diff_cpu = relative_difference(order_1_ref_cpu, order_1_cpu.numpy())
    order_2_diff_cpu = relative_difference(order_2_ref_cpu, order_2_cpu.numpy())

    x_gpu = x_cpu.cuda()
    order_0_gpu = compute_integrals(x_gpu, integral_powers)
    order_1_gpu, order_2_gpu = scattering(x_gpu, order_2=True,
                            method='integral', integral_powers=integral_powers)
    order_0_diff_gpu = relative_difference(order_0_ref_gpu, order_0_gpu.cpu().numpy())
    order_1_diff_gpu = relative_difference(order_1_ref_gpu, order_1_gpu.cpu().numpy())
    order_2_diff_gpu = relative_difference(order_2_ref_gpu, order_2_gpu.cpu().numpy())

    assert  order_0_diff_cpu < 1e-6, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
    assert  order_1_diff_cpu < 1e-6, "CPU : order 1 do not match, diff={}".format(order_1_diff_cpu)
    assert  order_2_diff_cpu < 1e-6, "CPU : order 2 do not match, diff={}".format(order_2_diff_cpu)

    assert  order_0_diff_gpu < 1e-6, "GPU : order 0 do not match, diff={}".format(order_0_diff_gpu)
    assert  order_1_diff_gpu < 1e-6, "GPU : order 1 do not match, diff={}".format(order_1_diff_gpu)
    assert  order_2_diff_gpu < 1e-6, "GPU : order 2 do not match, diff={}".format(order_2_diff_gpu)


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
