import torch.nn as nn
import torch.nn.functional as F


class Conv1DFFT(nn.Module):
    def __init__(self, backend):
        super(Conv1dFFT, self).__init__()
        self.name = 'torch'
        self.backend = backend

    def preprocess_signal(self, x):
        x_hat = self.backend.rfft(x)
        return x_hat

    def convolution(self, x_hat, conv_filter, subsampling_factor, domain=''):
        y_c = self.backend.cdgmm(x_hat, conv_filter)
        y_hat = self.backend.subsample_fourier(y_c, 2 ** subsampling_factor)
        if domain == 'real':
            y_r = self.backend.irfft(y_hat)
        else:
            y_r = self.backend.ifft(y_hat)
        return y_r
