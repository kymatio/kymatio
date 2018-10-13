""" This script will test the submodules used by the scattering module"""

import torch
from scattering.scattering2d import Scattering2D
from scattering.scattering2d import utils as sl


def linfnorm(x,y):
    return torch.max(torch.abs(x-y))

def test_FFTCentralFreq(gpu=False):
    # Checked the 0 frequency
    x = torch.FloatTensor(10, 10, 2).fill_(0)
    x.narrow(2, 0, 1).fill_(1)
    if gpu:
        x = x.cuda()

    a = x.sum()
    fft = sl.Fft()
    fft(x, inplace=True)
    b = x[0,0,0]
    assert (a - b).abs().max() < 1e-6

def test_FFTCentralFreqBatch(gpu=False):
    # Same for batches
    x = torch.FloatTensor(3,10,10,2).fill_(0)
    x.narrow(3,0,1).fill_(1)
    if gpu:
        x = x.cuda()

    a = x.sum()
    fft = sl.Fft()
    fft(x, inplace=True)
    c = x[:,0,0,0].sum()
    assert (a - c).abs().max() < 1e-6


def test_FFTUnormalized(gpu=False):
    # Check for a random tensor:
    x = torch.FloatTensor(25, 17, 3, 2).bernoulli_(0.5)

    if gpu:
        x=x.cuda()
    else:
        x=x.cpu()
    x.narrow(3,1,1).fill_(0)

    fft=sl.Fft()
    y = fft(x)
    z = fft(y, direction='C2R')

    z /= 17*3 # FFTs are unnormalized


    assert (linfnorm(x.select(3,0), z) - 0).abs().max() < 1e-6




# Checkked the modulus
def test_Modulus():
    for jit in [True, False]:
        modulus = sl.Modulus(jit=jit)
        x = torch.cuda.FloatTensor(100,10,4,2).copy_(torch.rand(100,10,4,2))
        y = modulus(x)
        u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
        v = y.narrow(3, 0, 1)

        assert (u - v).abs().max() < 1e-6


def test_Periodization():
    for jit in [True, False]:
        x = torch.rand(100, 1, 128, 128, 2).cuda().double()
        y = torch.zeros(100, 1, 8, 8, 2).cuda().double()

        for i in range(8):
            for j in range(8):
                for m in range(16):
                    for n in range(16):
                        y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

        y = y / (16*16)

        periodize = sl.Periodize(jit=jit)

        z = periodize(x, k=16)
        assert (y - z).abs().max() < 1e-8

        z = periodize(x.cpu(), k=16)
        assert (y.cpu() - z).abs().max() < 1e-8


# Check the CUBLAS routines
def test_Cublas():
    for jit in [True, False]:
        x = torch.rand(100,128,128,2).cuda()
        filter = torch.rand(128,128,2).cuda()
        filter[..., 1] = 0
        y = torch.ones(100,128,128,2).cuda()
        z = torch.Tensor(100,128,128,2).cuda()

        for i in range(100):
            y[i,:,:,0]=x[i,:,:,0] * filter[:,:,0]-x[i,:,:,1] * filter[:,:,1]
            y[i, :, :, 1] = x[i, :, :, 1] * filter[:, :, 0] + x[i, :, :, 0] *filter[:, :, 1]
        z = sl.cdgmm(x, filter, jit=jit)

        assert (y-z).abs().max() < 1e-6

def test_Scattering2D():
    data = torch.load('test_data.pt')
    x = data['x'].view([7,3, 128, 128])
    S = data['S'].view([7,3, 417, 8, 8])

    # First, let's check the Jit
    scat = Scattering2D(128, 128, 4, pre_pad=False,jit=True)
    scat.cuda()
    x = x.cuda()
    S = S.cuda()
    y = scat(x)
    assert ((S - y)).abs().max() < 1e-6

    # Then, let's check when using pure pytorch code
    scat = Scattering2D(128, 128, 4, pre_pad=False, jit=False)
    Sg = []
    Sc = []
    for gpu in [True, False]:
        if gpu:
            x = x.cuda()
            scat.cuda()
            S = S.cuda()
            Sg = scat(x)
        else:
            x = x.cpu()
            S = S.cpu()
            scat.cpu()
            Sg = scat(x)
            """there are huge round off errors with fftw, numpy fft, cufft...
            and the kernels of periodization. We do not wish to play with that as it is meaningless."""
        print((Sg - S).abs().max())
        assert (Sg-S).abs().max() < 1e-1



