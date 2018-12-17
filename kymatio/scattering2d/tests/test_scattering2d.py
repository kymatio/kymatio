""" This script will test the submodules used by the scattering module"""

import os
import numpy as np
import torch
import pytest
from kymatio.scattering2d import Scattering2D
from kymatio.scattering2d import backend


backends = []
try:
    from kymatio.scattering2d.backend import backend_skcuda
    backends.append(backend_skcuda)
except:
    pass

try:
    from kymatio.scattering2d.backend import backend_torch
    backends.append(backend_torch)
except:
    pass

if torch.cuda.is_available():
    devices = ['gpu', 'cpu']
else:
    devices = ['cpu']


# Checked the modulus
def test_Modulus():
    for device in devices:
        if device == 'gpu':
            for backend in backends:
                modulus = backend.Modulus()
                x = torch.rand(100, 10, 4, 2).cuda().float()
                y = modulus(x)
                u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
                v = y.narrow(3, 0, 1)
                u = u.squeeze()
                v = v.squeeze()
                assert (u - v).abs().max() < 1e-6
        elif device == 'cpu':
            for backend in backends:
                if backend.NAME == 'skcuda':
                    continue
                modulus = backend.Modulus()
                x = torch.rand(100, 10, 4, 2).float()
                y = modulus(x)
                u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
                v = y.narrow(3, 0, 1)
                u = u.squeeze()
                v = v.squeeze()
                assert (u - v).abs().max() < 1e-6
        else:
            raise('No backend or device detected.')



# Checked the subsampling
def test_SubsampleFourier():
    for device in devices:
        if device == 'gpu':
            for backend in backends:
                x = torch.rand(100, 1, 128, 128, 2).cuda().double()
                y = torch.zeros(100, 1, 8, 8, 2).cuda().double()

                for i in range(8):
                    for j in range(8):
                        for m in range(16):
                            for n in range(16):
                                y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

                y = y / (16*16)

                subsample_fourier = backend.SubsampleFourier()

                z = subsample_fourier(x, k=16)
                assert (y - z).abs().max() < 1e-8
                if backend.NAME == 'torch':
                    z = subsample_fourier(x.cpu(), k=16)
                    assert (y.cpu() - z).abs().max() < 1e-8
        elif device == 'cpu':
            for backend in backends:
                if backend.NAME == 'skcuda':
                    continue
                x = torch.rand(100, 1, 128, 128, 2).double()
                y = torch.zeros(100, 1, 8, 8, 2).double()

                for i in range(8):
                    for j in range(8):
                        for m in range(16):
                            for n in range(16):
                                y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

                y = y / (16*16)

                subsample_fourier = backend.SubsampleFourier()

                z = subsample_fourier(x, k=16)
                assert (y - z).abs().max() < 1e-8
                if backend.NAME == 'torch':
                    z = subsample_fourier(x.cpu(), k=16)
                    assert (y.cpu() - z).abs().max() < 1e-8
        else:
            raise ('No backend or device detected.')


# Check the CUBLAS routines
def test_Cublas():
    for device in devices:
        if device == 'gpu':
            for backend in backends:
                x = torch.rand(100, 128, 128, 2).cuda()
                filter = torch.rand(128, 128, 2).cuda()
                filter[..., 1] = 0
                y = torch.ones(100, 128, 128, 2).cuda()
                z = torch.Tensor(100, 128, 128, 2).cuda()

                for i in range(100):
                    y[i,:,:,0]=x[i,:,:,0] * filter[:,:,0]-x[i,:,:,1] * filter[:,:,1]
                    y[i, :, :, 1] = x[i, :, :, 1] * filter[:, :, 0] + x[i, :, :, 0] * filter[:, :, 1]
                z = backend.cdgmm(x, filter)

                assert (y-z).abs().max() < 1e-6
        elif device == 'cpu':
            for backend in backends:
                if backend.NAME == 'skcuda':
                    continue
                x = torch.rand(100, 128, 128, 2)
                filter = torch.rand(128, 128, 2)
                filter[..., 1] = 0
                y = torch.ones(100, 128, 128, 2)
                z = torch.Tensor(100, 128, 128, 2)

                for i in range(100):
                    y[i, :, :, 0] = x[i, :, :, 0] * filter[:, :, 0] - x[i, :, :, 1] * filter[:, :, 1]
                    y[i, :, :, 1] = x[i, :, :, 1] * filter[:, :, 0] + x[i, :, :, 0] * filter[:, :, 1]
                z = backend.cdgmm(x, filter)

                assert (y - z).abs().max() < 1e-6


def reorder_coefficients_from_interleaved(J, L):
    # helper function to obtain positions of order0, order1, order2 from interleaved
    order0, order1, order2 = [], [], []
    n_order0, n_order1, n_order2 = 1, J * L, L ** 2 * J * (J - 1) // 2
    n = 0
    order0.append(n)
    for j1 in range(J):
        for l1 in range(L):
            n += 1
            order1.append(n)
            for j2 in range(j1 + 1, J):
                for l2 in range(L):
                    n += 1
                    order2.append(n)
    assert len(order0) == n_order0
    assert len(order1) == n_order1
    assert len(order2) == n_order2
    return order0, order1, order2


# Check the scattering
# FYI: access the two different tests in here by setting envs
# KYMATIO_BACKEND=skcuda and KYMATIO_BACKEND=torch
def test_Scattering2D():
    test_data_dir = os.path.dirname(__file__)
    data = torch.load(os.path.join(test_data_dir, 'test_data_2d.pt'))

    x = data['x']
    S = data['Sx']
    J = data['J']

    # we need to reorder S from interleaved (how it's saved) to o0, o1, o2
    # (which is how it's now computed)

    o0, o1, o2 = reorder_coefficients_from_interleaved(J, L=8)
    reorder = torch.from_numpy(np.concatenate((o0, o1, o2)))
    S = S[..., reorder, :, :]

    pre_pad = data['pre_pad']

    M = x.shape[2]
    N = x.shape[3]

    import kymatio.scattering2d.backend as backend

    if backend.NAME == 'skcuda':
        print('skcuda backend tested!')
        # First, let's check the Jit
        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad)
        scattering.cuda()
        x = x.cuda()
        S = S.cuda()
        y = scattering(x)
        assert ((S - y)).abs().max() < 1e-6
    elif backend.NAME == 'torch':
        # Then, let's check when using pure pytorch code
        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad)
        Sg = []

        for device in devices:
            if device == 'gpu':
                print('torch-gpu backend tested!')
                x = x.cuda()
                scattering.cuda()
                S = S.cuda()
                Sg = scattering(x)
            else:
                print('torch-cpu backend tested!')
                x = x.cpu()
                S = S.cpu()
                scattering.cpu()
                Sg = scattering(x)
            assert (Sg - S).abs().max() < 1e-6


def test_batch_shape_agnostic():
    J = 3
    L = 8
    shape = (32, 32)

    shape_ds = tuple(n // 2 **J for n in shape)

    S = Scattering2D(J, shape, L)

    with pytest.raises(RuntimeError) as ve:
        S(torch.zeros(()))
    assert "at least two" in ve.value.args[0]

    with pytest.raises(RuntimeError) as ve:
        S(torch.zeros((32, )))
    assert "at least two" in ve.value.args[0]

    x = torch.zeros(shape)

    if backend.NAME == 'skcuda':
        x = x.cuda()
        S.cuda()

    Sx = S(x)

    assert len(Sx.shape) == 3
    assert Sx.shape[-2:] == shape_ds

    n_coeffs = Sx.shape[-3]

    test_shapes = ((1,) + shape, (2,) + shape, (2, 2) + shape,
                   (2, 2, 2) + shape)

    for test_shape in test_shapes:
        x = torch.zeros(test_shape)

        if backend.NAME == 'skcuda':
            x = x.cuda()

        Sx = S(x)

        assert len(Sx.shape) == len(test_shape) + 1
        assert Sx.shape[-2:] == shape_ds
        assert Sx.shape[-3] == n_coeffs
        assert Sx.shape[:-3] == test_shape[:-2]
