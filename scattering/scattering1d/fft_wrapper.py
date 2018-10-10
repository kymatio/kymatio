import torch

def fft1d_c2c(x):
    return torch.fft(x,1,normalized=False)


def ifft1d_c2c(x):
    return torch.ifft(x,1,normalized=False)


def ifft1d_c2c_normed(x):
    return torch.ifft(x, 1, normalized=True)
