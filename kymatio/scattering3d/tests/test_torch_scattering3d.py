""" This script will test the submodules used by the scattering module"""
import torch
import os
import numpy as np
import pytest
from kymatio import HarmonicScattering3D
from kymatio.scattering3d import backend
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians


backends = []

try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        from kymatio.scattering3d.backend.torch_skcuda_backend import backend
        backends.append(backend)
except:
    pass

try:
    from kymatio.scattering3d.backend.torch_backend import backend
    backends.append(backend)
except:
    pass


if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']



def relative_difference(a, b):
    return np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), np.sum(np.abs(b)))

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_FFT3d_central_freq_batch(device, backend):
    # Checked the 0 frequency for the 3D FFT
    for device in devices:
        x = torch.zeros(1, 32, 32, 32, 2).float()
        if device == 'gpu':
            x = x.cuda()
        a = x.sum()
        y = backend.fft(x)
        c = y[:,0,0,0].sum()
        assert (c-a).abs().sum()<1e-6

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_fft3d_error(backend, device):
    x = torch.zeros(8, 1)
    with pytest.raises(TypeError) as record:
        backend.fft(x)
    assert "should be complex" in record.value.args[0]

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("inplace", [False, True])
def test_cdgmm3d(device, backend, inplace):
    # Not all backends currently implement the inplace variant
    x = torch.zeros(2, 3, 4, 2).to(device)
    x[...,0] = 2
    x[...,1] = 3

    y = torch.zeros_like(x)
    y[...,0] = 4
    y[...,1] = 5

    prod = torch.zeros_like(x)
    prod[...,0] = x[...,0]*y[...,0] - x[...,1]*y[...,1]
    prod[...,1] = x[...,0]*y[...,1] + x[...,1]*y[...,0]

    z = backend.cdgmm3d(x, y, inplace=inplace)

    assert (z-prod).norm().cpu().item() < 1e-7

    if inplace:
        assert (x-z).norm().cpu().item() < 1e-7

    with pytest.warns(UserWarning) as record:
        x = torch.randn((3, 4, 3, 2), device=device)
        x = x[:,0:3,...]
        y = torch.randn((3, 3, 3, 2), device=device)
        backend.cdgmm3d(x, y)
    assert "A is converted" in record[0].message.args[0]

    with pytest.warns(UserWarning) as record:
        x = torch.randn((3, 3, 3, 2), device=device)
        y = torch.randn((3, 4, 3, 2), device=device)
        y = y[:,0:3,...]
        backend.cdgmm3d(x, y)
    assert "B is converted" in record[0].message.args[0]

    with pytest.raises(RuntimeError) as record:
        x = torch.randn((3, 3, 3, 2), device=device)
        y = torch.randn((4, 4, 4, 2), device=device)
        backend.cdgmm3d(x, y)
    assert "not compatible" in record.value.args[0]

    x = torch.randn((2, 3, 3, 3, 2), device=device)
    y = torch.randn((3, 3, 3, 2), device=device)
    backend.cdgmm3d(x, y)

    with pytest.raises(TypeError) as record:
        x = torch.randn((3, 3, 3, 1), device=device)
        y = torch.randn((3, 3, 3, 1), device=device)
        backend.cdgmm3d(x, y)
    assert "should be complex" in record.value.args[0]

    # This one is a little tricky. We can't have the number of dimensions be
    # greater than 4 since that triggers the "not compatible" error.
    with pytest.raises(RuntimeError) as record:
        x = torch.randn((3, 3, 2), device=device)
        y = torch.randn((3, 3, 2), device=device)
        backend.cdgmm3d(x, y)
    assert "must be simply a complex" in record.value.args[0]

    # Create a tensor that behaves like `torch.Tensor` but is technically a
    # different type.
    class FakeTensor(torch.Tensor):
        pass

    with pytest.raises(RuntimeError) as record:
        x = FakeTensor(3, 3, 3, 2)
        y = torch.randn(3, 3, 3, 2)
        backend.cdgmm3d(x, y)
    assert "should be same type" in record.value.args[0]

    if backend.name == 'torch_skcuda':
        x = torch.randn((3, 3, 3, 2), device=torch.device('cpu'))
        y = torch.randn((3, 3, 3, 2), device=torch.device('cpu'))
        with pytest.raises(RuntimeError) as record:
            backend.cdgmm3d(x, y)
        assert "for cpu tensors" in record.value.args[0]

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_iscomplex(backend, device):
    assert backend.iscomplex(torch.zeros(4, 2).to(device))
    assert not backend.iscomplex(torch.zeros(4, 1).to(device))

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_to_complex(backend, device):
    x = torch.randn(4, 3).to(device)
    xc = backend.to_complex(x)
    assert (xc[...,0] == x).all()
    assert (xc[...,1] == 0).all()

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_complex_modulus(backend, device):
    x = torch.randn(4, 3, 2).to(device)
    xm = torch.sqrt(x[...,0] ** 2 + x[...,1] ** 2)
    y = backend.complex_modulus(x)
    assert (y[...,0] - xm).norm() < 1e-7
    assert (y[...,1]).norm() < 1e-7

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_against_standard_computations(device, backend):
    file_path = os.path.abspath(os.path.dirname(__file__))
    data = torch.load(os.path.join(file_path, 'test_data_3d.pt'))
    x = data['x'].to(device)
    scattering_ref = data['Sx'].to(device)
    J = data['J']
    L = data['L']
    integral_powers = data['integral_powers']

    M = x.shape[1]

    batch_size = x.shape[0]

    N, O = M, M
    sigma = 1

    scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma, backend=backend, frontend='torch').to(device)



    order_0 = backend.compute_integrals(x, integral_powers)
    scattering.max_order = 2
    scattering.method = 'integral'
    scattering.integral_powers = integral_powers
    orders_1_and_2 = scattering(x)

    # WARNING: These are hard-coded values for the setting J = 2.
    n_order_1 = 3
    n_order_2 = 3

    # Extract orders and make order axis the slowest in accordance with
    # the stored reference scattering transform.
    order_1 = orders_1_and_2[:,0:n_order_1,...]
    order_2 = orders_1_and_2[:,n_order_1:n_order_1+n_order_2,...]

    # Permute the axes since reference has (batch index, integral power, j,
    # ell) while the computed transform has (batch index, j, ell, integral
    # power).
    order_1 = order_1.permute(0, 3, 1, 2)
    order_2 = order_2.permute(0, 3, 1, 2)

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

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_solid_harmonic_scattering(device, backend):
    # Compare value to analytical formula in the case of a single Gaussian
    centers = torch.FloatTensor(1, 1, 3).fill_(0).to(device)
    weights = torch.FloatTensor(1, 1).fill_(1).to(device)
    sigma_gaussian = 3.
    sigma_0_wavelet = 3.
    M, N, O, J, L = 128, 128, 128, 1, 3
    grid = torch.from_numpy(
        np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
    x = generate_weighted_sum_of_gaussians(grid, centers, weights, sigma_gaussian)
    scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma_0_wavelet).to(device)

    scattering.max_order = 1
    scattering.method = 'integral'
    scattering.integral_powers = [1]


    s = scattering(x)

    for j in range(J+1):
        sigma_wavelet = sigma_0_wavelet*2**j
        k = sigma_wavelet / np.sqrt(sigma_wavelet**2 + sigma_gaussian**2)
        for l in range(1, L+1):
            err = torch.abs(s[0, j, l, 0] - k ** l).sum()/(1e-6+s[0, j, l, 0].abs().sum())
            assert err<1e-4

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_larger_scales(device, backend):
    shape = (32, 32, 32)
    L = 3
    sigma_0 = 1

    x = torch.randn((1,) + shape).to(device)

    for J in range(3, 4+1):
        scattering = HarmonicScattering3D(J=J, shape=shape, L=L, sigma_0=sigma_0, frontend='torch', backend=backend).to(device)
        scattering.method = 'integral'
        Sx = scattering(x)

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_scattering_methods(device, backend):
    shape = (32, 32, 32)
    J = 4
    L = 3
    sigma_0 = 1
    x = torch.randn((1,) + shape).to(device)

    scattering = HarmonicScattering3D(J=J, shape=shape, L=L, sigma_0=sigma_0).to(device)

    scattering.method = 'standard'
    Sx = scattering(x)
    scattering.rotation_covariant = False
    Sx = scattering(x)

    points = torch.zeros(1, 1, 3)
    points[0,0,:] = torch.tensor(shape)/2

    scattering.method = 'local'
    scattering.points = points
    Sx = scattering(x)
    scattering.rotation_covariant = False
    Sx = scattering(x)

