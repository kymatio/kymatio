import torch.nn as nn
import torch.nn.functional as F

class Conv1dFFTPrimative(nn.Module):
    def __init__(self, backend):
        super(Conv1dFFTPrimative, self).__init__()
        self.name = 'torch'
        self.backend = backend

    def preprocess_signal(self, x, pad_left=None, pad_right=None):
        if pad_left is not None:
            x = self.backend.pad(x, pad_left=pad_left, pad_right=pad_right)
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


class Conv1dSpatialPrimative(nn.Module):
    def __init__(self, backend):
        super(Conv1dSpatialPrimative, self).__init__()
        self.name = 'torch'
        self.backend = backend
    
    def forward(self, x, direction, conv_filter, direction_inverse, sampling_factor):
        print(x.shape, conv_filter.shape, "yes")
        y = F.conv1d(x, conv_filter)
        return x, y

