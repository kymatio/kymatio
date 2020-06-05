import torch.nn as nn
import torch.nn.functional as F

class Conv1dFFTPrimative(nn.Module):
    def __init__(self, backend):
        super(Conv1dFFTPrimative, self).__init__()
        self.name = 'torch'
        self.backend = backend
    
    def forward(self, x, direction, conv_filter, direction_inverse, sampling_factor):
        if direction != '':
            x_hat = self.backend.fft(x, direction)
        else:
            x_hat = x

        y_c = self.backend.cdgmm(x_hat, conv_filter)
        y_hat = self.backend.subsample_fourier(y_c, 2**sampling_factor)
        y_r = self.backend.fft(y_hat, direction_inverse, inverse=True)

        if direction == '':
            return y_hat, y_r
        else:
            return x_hat, y_r

class Conv1dSpatialPrimative(nn.Module):
    def __init__(self, backend):
        super(Conv1dSpatialPrimative, self).__init__()
        self.name = 'torch'
        self.backend = backend
    
    def forward(self, x, direction, conv_filter, direction_inverse, sampling_factor):
        print(x.shape, conv_filter.shape, "yes")
        y = F.conv1d(x, conv_filter)
        return x, y

