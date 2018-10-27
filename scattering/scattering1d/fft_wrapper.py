import torch

"""
Package used to wrap the CUFFT

The main function to use at the end are fft1d_c2c and ifft1d_c2c_normed
which take as input torch Variable objects and return Variable containing
the FFT, which are differentiable with respect to their inputs.
"""


def fft1d_c2c(x):
    return torch.fft(x)


def ifft1d_c2c(x):
    return torch.ifft(x)


def ifft1d_c2c_normed(x):
    factor = 1. / float(x.shape[-2])
    return torch.ifft(x) * factor
