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
      
def relative_difference(a, b):
    return np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), np.sum(np.abs(b)))


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

def test_against_standard_computations():
    file_path = os.path.abspath(os.path.dirname(__file__))
    d = np.load(os.path.join(file_path, 'test_data_3d.pt'))
    x = d.item()['x']
    scat_ref = d.item()['scat']

    M, N, O, J, L, sigma = 64, 64, 64, 2, 2, 1.
    integral_powers = [1., 2.]

    scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

    x_cpu = torch.from_numpy(x)

    order_0_cpu = compute_integrals(x_cpu, integral_powers)
    order_1_cpu, order_2_cpu = scattering(x_cpu, order_2=True,
                            method='integral', integral_powers=integral_powers)


    order_0_cpu = order_0_cpu.numpy().reshape((1, -1))
    start = 0
    end = order_0_cpu.shape[1]
    order_0_ref = scat_ref[:,start:end]

    order_1_cpu = order_1_cpu.numpy().reshape((1, -1))
    start = end
    end += order_1_cpu.shape[1]
    order_1_ref = scat_ref[:, start:end]

    order_2_cpu = order_2_cpu.numpy().reshape((1, -1))
    start = end
    end += order_2_cpu.shape[1]
    order_2_ref = scat_ref[:, start:end]

    order_0_diff_cpu = relative_difference(order_0_ref, order_0_cpu)
    order_1_diff_cpu = relative_difference(order_1_ref, order_1_cpu)
    order_2_diff_cpu = relative_difference(order_2_ref, order_2_cpu)

    x_gpu = x_cpu.cuda()
    order_0_gpu = compute_integrals(x_gpu, integral_powers)
    order_1_gpu, order_2_gpu = scattering(x_gpu, order_2=True,
                            method='integral', integral_powers=integral_powers)
    order_0_gpu = order_0_gpu.cpu().numpy().reshape((1, -1))
    order_1_gpu = order_1_gpu.cpu().numpy().reshape((1, -1))
    order_2_gpu = order_2_gpu.cpu().numpy().reshape((1, -1))

    order_0_diff_gpu = relative_difference(order_0_ref, order_0_gpu)
    order_1_diff_gpu = relative_difference(order_1_ref, order_1_gpu)
    order_2_diff_gpu = relative_difference(order_2_ref, order_2_gpu)

    assert  order_0_diff_cpu < 1e-6, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
    assert  order_1_diff_cpu < 1e-6, "CPU : order 1 do not match, diff={}".format(order_1_diff_cpu)
    assert  order_2_diff_cpu < 1e-6, "CPU : order 2 do not match, diff={}".format(order_2_diff_cpu)

    assert  order_0_diff_gpu < 1e-6, "GPU : order 0 do not match, diff={}".format(order_0_diff_gpu)
    assert  order_1_diff_gpu < 1e-6, "GPU : order 1 do not match, diff={}".format(order_1_diff_gpu)
    assert  order_2_diff_gpu < 1e-6, "GPU : order 2 do not match, diff={}".format(order_2_diff_gpu)
    