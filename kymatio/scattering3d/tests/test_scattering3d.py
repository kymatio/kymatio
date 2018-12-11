""" This script will test the submodules used by the scattering module"""
import torch
import os
import warnings
import numpy as np
import pytest
from kymatio import Scattering3D
from kymatio.scattering3d import backend
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians, compute_integrals, sqrt

if torch.cuda.is_available():
    devices = ['gpu', 'cpu']
else:
    devices = ['cpu']

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
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend is not yet implemented for 3D "
            "scattering, but that's ok (for now)."), RuntimeWarning,
            stacklevel=2)
        return

    # Checked the 0 frequency for the 3D FFT
    for device in devices:
        x = torch.zeros(1, 32, 32, 32, 2).float()
        if device == 'gpu':
            x = x.cuda()
        a = x.sum()
        y = backend.fft(x)
        c = y[:,0,0,0].sum()
        assert (c-a).abs().sum()<1e-6


def test_against_standard_computations():
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend is not yet implemented for 3D "
            "scattering, but that's ok (for now)."), RuntimeWarning,
            stacklevel=2)
        return

    file_path = os.path.abspath(os.path.dirname(__file__))
    data = torch.load(os.path.join(file_path, 'test_data_3d.pt'))
    x = data['x']
    scattering_ref = data['Sx']
    J = data['J']
    L = data['L']
    integral_powers = data['integral_powers']

    M = x.shape[1]

    batch_size = x.shape[0]

    N, O = M, M
    sigma = 1

    scattering = Scattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma)

    for device in devices:
        if device == 'cpu':
            x = x.cpu()
            scattering.cpu()
        else:
            x = x.cuda()
            scattering.cuda()
        order_0 = compute_integrals(x, integral_powers)
        orders_1_and_2 = scattering(
            x, order_2=True, method='integral',
            integral_powers=integral_powers)

        # WARNING: These are hard-coded values for the setting J = 2.
        n_order_1 = 3
        n_order_2 = 3

        # Extract orders and make order axis the slowest in accordance with
        # the stored reference scattering transform.
        order_1 = orders_1_and_2[:,:,0:n_order_1,:]
        order_2 = orders_1_and_2[:,:,n_order_1:n_order_1+n_order_2,:]

        order_1 = order_1.reshape((batch_size, -1))
        order_2 = order_2.reshape((batch_size, -1))

        orders_1_and_2 = torch.cat((order_1, order_2), 1)

        order_0 = order_0.cpu().numpy().reshape((batch_size, -1))
        start = 0
        end = order_0.shape[1]
        order_0_ref = scattering_ref[:,start:end].numpy()

        orders_1_and_2 = orders_1_and_2.cpu().numpy().reshape((batch_size, -1))
        start = end
        end += orders_1_and_2.shape[1]
        orders_1_and_2_ref = scattering_ref[:, start:end].numpy()

        order_0_diff_cpu = relative_difference(order_0_ref, order_0)
        orders_1_and_2_diff_cpu = relative_difference(
            orders_1_and_2_ref, orders_1_and_2)

        assert order_0_diff_cpu < 1e-6, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
        assert orders_1_and_2_diff_cpu < 1e-6, "CPU : orders 1 and 2 do not match, diff={}".format(orders_1_and_2_diff_cpu)


def test_solid_harmonic_scattering():
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend is not yet implemented for 3D "
            "scattering, but that's ok (for now)."), RuntimeWarning,
            stacklevel=2)
        return

    # Compare value to analytical formula in the case of a single Gaussian
    centers = torch.FloatTensor(1, 1, 3).fill_(0)
    weights = torch.FloatTensor(1, 1).fill_(1)
    sigma_gaussian = 3.
    sigma_0_wavelet = 3.
    M, N, O, J, L = 128, 128, 128, 1, 3
    grid = torch.from_numpy(
        np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
    x = generate_weighted_sum_of_gaussians(grid, centers, weights, sigma_gaussian)
    scattering = Scattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma_0_wavelet)
    s = scattering(x, order_2=False, method='integral', integral_powers=[1])

    for j in range(J+1):
        sigma_wavelet = sigma_0_wavelet*2**j
        k = sigma_wavelet / np.sqrt(sigma_wavelet**2 + sigma_gaussian**2)
        for l in range(1, L+1):
            err = torch.abs(s[0, 0, j, l] - k ** l).sum()/(1e-6+s[0, 0, j, l].abs().sum())
            assert err<1e-4

def test_larger_scales():
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend is not yet implemented for 3D "
            "scattering, but that's ok (for now)."), RuntimeWarning,
            stacklevel=2)
        return

    shape = (32, 32, 32)
    L = 3
    sigma_0 = 1

    x = torch.randn((1,) + shape)

    for J in range(3, 4+1):
        scattering = Scattering3D(J=J, shape=shape, L=L, sigma_0=sigma_0)
        Sx = scattering(x, method='integral')

def test_scattering_methods():
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend is not yet implemented for 3D "
            "scattering, but that's ok (for now)."), RuntimeWarning,
            stacklevel=2)
        return

    shape = (32, 32, 32)
    J = 4
    L = 3
    sigma_0 = 1
    x = torch.randn((1,) + shape)

    scattering = Scattering3D(J=J, shape=shape, L=L, sigma_0=sigma_0)
    Sx = scattering(x, method='standard')
    Sx = scattering(x, method='standard', rotation_covariant=False)

    points = torch.zeros(1, 1, 3)
    points[0,0,:] = torch.tensor(shape)/2

    Sx = scattering(x, method='local', points=points)
    Sx = scattering(x, method='local', points=points, rotation_covariant=False)

def test_cpu_cuda():
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend is not yet implemented for 3D "
            "scattering, but that's ok (for now)."), RuntimeWarning,
            stacklevel=2)
        return

    shape = (32, 32, 32)
    J = 4
    L = 3
    sigma_0 = 1
    x = torch.randn((1,) + shape)

    S = Scattering3D(J=J, shape=shape, L=L, sigma_0=sigma_0)

    assert not S.is_cuda

    Sx = S(x)

    if 'gpu' in devices:
        x_gpu = x.cuda()

        with pytest.raises(TypeError) as record:
            Sx_gpu = S(x_gpu)
        assert "is in CPU mode" in record.value.args[0]

        S.cuda()

        assert S.is_cuda

        Sx_gpu = S(x_gpu)

        assert Sx_gpu.is_cuda

        with pytest.raises(TypeError) as record:
            Sx = S(x)
        assert "is in GPU mode" in record.value.args[0]

    S.cpu()

    assert not S.is_cuda

def test_utils():
    # Simple test
    x = np.arange(8)
    y = sqrt(x**2)
    assert (y == x).all()

    # Test problematic case
    x = np.arange(8193, dtype='float32')
    y = sqrt(x**2)
    assert (y == x).all()

    # Make sure we still don't let in negatives...
    with pytest.warns(RuntimeWarning) as record:
        x = np.array([-1, 0, 1])
        y = sqrt(x)
    assert "Negative" in record[0].message.args[0]

    # ...unless they are compled numbers!
    with pytest.warns(None) as record:
        x = np.array([-1, 0, 1], dtype='complex64')
        y = sqrt(x)
    assert not record.list
