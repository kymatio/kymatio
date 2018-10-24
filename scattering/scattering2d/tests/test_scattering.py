""" This script will test the submodules used by the scattering module"""

import os
import torch
from scattering.scattering2d import Scattering2D
from scattering.scattering2d import utils as sl


if CUDA_AVAILABLE:
    gpus = [True, False]
    backends = ['pytorch', 'skcuda']
else:
    gpus = [False]
    backends = ['pytorch']
# Checked the modulus
def test_Modulus():
    for backend in backends:
        modulus = sl.Modulus(backend=backend)
        x = torch.rand(100, 10, 4, 2).float()
        if CUDA_AVAILABLE:
            x =  x.cuda()

        y = modulus(x)
        u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
        v = y.narrow(3, 0, 1)
        u = u.squeeze()
        v = v.squeeze()
        assert (u - v).abs().max() < 1e-6


def test_Periodization():
    for backend in backends:
        x = torch.rand(100, 1, 128, 128, 2).double()
        y = torch.zeros(100, 1, 8, 8, 2).double()

        if CUDA_AVAILABLE:
            x = x.cuda()
            y = y.cuda()

        for i in range(8):
            for j in range(8):
                for m in range(16):
                    for n in range(16):
                        y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

        y = y / (16*16)

        periodize = sl.Periodize(backend=backend)

        z = periodize(x, k=16)
        assert (y - z).abs().max() < 1e-8
        if backend == 'torch':
            z = periodize(x.cpu(), k=16)
            assert (y.cpu() - z).abs().max() < 1e-8


# Check the CUBLAS routines
def test_Cublas():
    for backend in backends:
        x = torch.rand(100, 128, 128, 2)
        filter = torch.rand(128, 128, 2)
        filter[..., 1] = 0
        y = torch.ones(100, 128, 128, 2)
        if CUDA_AVAILABLE:
            x = x.cuda()
            filter = filter.cuda()
            y = y.cuda()

        for i in range(100):
            y[i,:,:,0]=x[i,:,:,0] * filter[:,:,0]-x[i,:,:,1] * filter[:,:,1]
            y[i, :, :, 1] = x[i, :, :, 1] * filter[:, :, 0] + x[i, :, :, 0] * filter[:, :, 1]
        z = sl.cdgmm(x, filter, backend=backend)

        assert (y-z).abs().max() < 1e-6

# Check the scattering
def test_Scattering2D():

    test_data_dir = os.path.dirname(__file__)
    data = torch.load(os.path.join(test_data_dir, 'test_data.pt'))
    x = data['x'].view(7, 3, 128, 128)
    S = data['S'].view(7, 3, 417, 8, 8)

    if 'skcuda' in backends:
        # First, let's check the Jit
        scattering = Scattering2D(128, 128, 4, pre_pad=False, backend='skcuda')
        scattering.cuda()
        x = x.cuda()
        S = S.cuda()
        y = scattering(x)
        assert ((S - y)).abs().max() < 1e-6

    # Then, let's check when using pure pytorch code
    scattering = Scattering2D(128, 128, 4, pre_pad=False, backend='torch')
    Sg = []

    for gpu in gpus:
        if gpu:
            x = x.cuda()
            scattering.cuda()
            S = S.cuda()
            Sg = scattering(x)
        else:
            x = x.cpu()
            S = S.cpu()
            scattering.cpu()
            Sg = scattering(x)
        assert (Sg - S).abs().max() < 1e-6

