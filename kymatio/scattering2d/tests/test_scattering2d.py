""" This script will test the submodules used by the scattering module"""

import os
import numpy as np
import torch
import pytest
from kymatio.scattering2d import Scattering2D
from kymatio.scattering2d import backend


if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


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

    scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad)
    Sg = []

    for device in devices:
        if device == 'cuda':
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

    Sx = S(x)

    assert len(Sx.shape) == 3
    assert Sx.shape[-2:] == shape_ds

    n_coeffs = Sx.shape[-3]

    test_shapes = ((1,) + shape, (2,) + shape, (2, 2) + shape,
                   (2, 2, 2) + shape)

    for test_shape in test_shapes:
        x = torch.zeros(test_shape)

        Sx = S(x)

        assert len(Sx.shape) == len(test_shape) + 1
        assert Sx.shape[-2:] == shape_ds
        assert Sx.shape[-3] == n_coeffs
        assert Sx.shape[:-3] == test_shape[:-2]

# Make sure we test for the errors that may be raised by
# `Scattering2D.forward`.
def test_scattering2d_errors():
    S = Scattering2D(3, (32, 32))

    with pytest.raises(TypeError) as record:
        S(None)
    assert('input should be' in record.value.args[0])

    x = torch.randn(4,4)
    y = x[::2,::2]

    with pytest.raises(RuntimeError) as record:
        S(y)
    assert('must be contiguous' in record.value.args[0])

    x = torch.randn(31, 31)

    with pytest.raises(RuntimeError) as record:
        S(x)
    assert('Tensor must be of spatial size' in record.value.args[0])

    S = Scattering2D(3, (32, 32), pre_pad=True)

    with pytest.raises(RuntimeError) as record:
        S(x)
    assert('Padded tensor must be of spatial size' in record.value.args[0])

    x = torch.randn(8,8)
    S = Scattering2D(2, (8, 8))

    for device in devices:
        x = x.to(device)
        S = S.to(device)
        y = S(x)
        assert(x.device == y.device)

# Check that several input size works
def test_input_size_agnostic():
    for N in [31, 32, 33]:
        for J in [1, 2, 4]:
            scattering = Scattering2D(J, shape=(N, N))
            x = torch.zeros(3, 3, N, N)

            S = scattering(x)
            scattering = Scattering2D(J, shape=(N, N), pre_pad=True)
            x = torch.zeros(3,3,scattering.M_padded, scattering.N_padded)

            S = scattering(x)

    N = 32
    J = 5
    scattering = Scattering2D(J, shape=(N, N))
    x = torch.zeros(3, 3, N, N)

    S = scattering(x)
    assert(S.shape[-2:] == (1, 1))

    N = 32
    J = 5
    scattering = Scattering2D(J, shape=(N+5, N))
    x = torch.zeros(3, 3, N+5, N)

    S = scattering(x)
    assert (S.shape[-2:] == (1, 1))
