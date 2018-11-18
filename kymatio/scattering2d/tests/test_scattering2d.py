""" This script will test the submodules used by the scattering module"""

import os
import torch
from scattering.scattering2d import Scattering2D
from scattering.scattering2d import backend


backends = []
try:
    from scattering.scattering2d.backend import backend_skcuda
    backends.append(backend_skcuda)
except:
    pass

try:
    from scattering.scattering2d.backend import backend_torch
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

# Check the scattering
# FYI: access the two different tests in here by setting envs
# SCATTERING_BACKEND=skcuda and SCATTERING_BACKEND=torch
def test_Scattering2D():
    test_data_dir = os.path.dirname(__file__)
    data = torch.load(os.path.join(test_data_dir, 'test_data_2d.pt'), map_location='cpu')
    x = data['x'].view(7, 3, 128, 128)
    S = data['S'].view(7, 3, 417, 8, 8)

    import scattering.scattering2d.backend as backend

    if backend.NAME == 'skcuda':
        print('skcuda backend tested!')
        # First, let's check the Jit
        scattering = Scattering2D(128, 128, 4, pre_pad=False)
        scattering.cuda()
        x = x.cuda()
        S = S.cuda()
        y = scattering(x)
        assert ((S - y)).abs().max() < 1e-6
    elif backend.NAME == 'torch':
        # Then, let's check when using pure pytorch code
        scattering = Scattering2D(128, 128, 4, pre_pad=False)
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

