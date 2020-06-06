import torch.nn as nn
import torch.nn.functional as F

class Conv1dFFTPrimative(nn.Module):
    def __init__(self, backend):
        super(Conv1dFFTPrimative, self).__init__()
        self.name = 'torch'
        self.backend = backend

    def preprocess_signal(self, x, direction):
        x_hat = self.backend.fft(x, direction)
        return x_hat

    def convolution(self, x_hat,  conv_filter, direction_inverse, sampling_factor):
        y_c = self.backend.cdgmm(x_hat, conv_filter)
        y_hat = self.backend.subsample_fourier(y_c, 2**sampling_factor)
        y_r = self.backend.fft(y_hat, direction_inverse, inverse=True)
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

