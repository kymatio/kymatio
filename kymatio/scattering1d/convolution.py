import torch.nn as nn
import torch.nn.functional as F

class Conv1dFFT(nn.Module):
    def __init__(self, backend):
        super(Conv1dFFT, self).__init__()
        self.name = 'torch'
        self.backend = backend

    def preprocess_signal(self, x, padding=()):
        if len(padding) != 0:
            x = self.backend.pad(x, pad_left=padding[0], pad_right=padding[1])
        x_hat = self.backend.rfft(x)
        return x_hat

    def convolution(self, x_hat, conv_filter, sampling_factor, domain='', ind_start=None, ind_end=None):
        y_c = self.backend.cdgmm(x_hat, conv_filter)
        y_hat = self.backend.subsample_fourier(y_c, 2**sampling_factor)
        if domain == 'real':
            y_r = self.backend.irfft(y_hat)
        else:
            y_r = self.backend.ifft(y_hat)

        if ind_start is not None:
            y_r = self.backend.unpad(y_r, ind_start, ind_end)

        return y_r
