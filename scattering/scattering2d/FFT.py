from __future__ import print_function
import torch
from torch.autograd import Function
from collections import defaultdict, namedtuple
from skcuda import cublas, cufft
import numpy as np


class FFTcache(object):

    def __init__(self):
        self.fft_cache=defaultdict(lambda: None)

    def buildCache(self, input, type):
        k = input.ndimension() - 3
        n = np.asarray([input.size(k), input.size(k+1)], np.int32)
        batch = input.nelement() // (2*input.size(k) * input.size(k + 1))
        idist = input.size(k) * input.size(k + 1)
        istride = 1
        ostride = istride
        odist = idist
        rank = 2
        print(rank, n.ctypes.data, n.ctypes.data, istride, idist, n.ctypes.data, ostride, odist, type, batch)
        plan = cufft.cufftPlanMany(rank, n.ctypes.data, n.ctypes.data, istride, idist, n.ctypes.data, ostride, odist, type, batch)
        self.fft_cache[(input.size(),type,input.get_device())] = plan

    def __del__(self):
        for keys in self.fft_cache:
            try:
                cufft.cufftDestroy(self.fft_cache[keys])
            except:
                pass

    def c2c(self, input, inverse=False):
        assert input.is_contiguous()
        output = input.new(input.size())
        flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
        if isinstance(input, torch.cuda.FloatTensor):
            ffttype = cufft.CUFFT_C2C
            exec_to_use = cufft.cufftExecC2C
        elif isinstance(input, torch.cuda.DoubleTensor):
            ffttype = cufft.CUFFT_Z2Z
            exec_to_use = cufft.cufftExecZ2Z
        else:
            raise TypeError('Unsupported type for input' + str(type(input)))
        if (self.fft_cache[(input.size(), ffttype, input.get_device())] is None):
            self.buildCache(input, ffttype)
        exec_to_use(self.fft_cache[(input.size(), ffttype, input.get_device())],
                    input.data_ptr(), output.data_ptr(), flag)
        return output

    def c2r(self, input):
        output = input.new(input.size()[:-1])
        if isinstance(input, torch.cuda.FloatTensor):
            ffttype = cufft.CUFFT_C2R
            exec_to_use = cufft.cufftExecC2R
        elif isinstance(input, torch.cuda.DoubleTensor):
            ffttype = cufft.CUFFT_Z2D
            exec_to_use = cufft.cufftExecZ2D
        else:
            raise TypeError('Unsupported type for input' + str(type(input)))
        if(self.fft_cache[(input.size(), ffttype, input.get_device())] is None):
            self.buildCache(input, ffttype)
        exec_to_use(self.fft_cache[(input.size(), ffttype, input.get_device())],
                    input.data_ptr(), output.data_ptr())
        return output

    def r2c(self, input):
        output = input.new(input.size() + torch.Size((2,)))
        if(self.fft_cache[(input.size(), cufft.CUFFT_R2C, input.get_device())] is None):
            self.buildCache(input, cufft.CUFFT_R2C)
        cufft.cufftExecR2C(self.fft_cache[(input.size(), cufft.CUFFT_R2C, input.get_device())],
                           input.data_ptr(), output.data_ptr())
        return output


fft = FFTcache()

class FFT_C2C(torch.autograd.Function):

    def forward(self, input):
        return fft.c2c(input, inverse=False)

    def backward(self, grad_output):
        return fft.c2c(grad_output, inverse=True)


class FFT_iC2C(torch.autograd.Function):

    def forward(self, input):
        return fft.c2c(input, inverse=True)

    def backward(self, grad_output):
        return fft.c2c(grad_output, inverse=False)

class FFT_C2R(torch.autograd.Function):

    def forward(self, input):
        return fft.c2r(input)

    def backward(self, grad_output):
        return fft.r2c(grad_output)


class FFT_R2C(torch.autograd.Function):

    def forward(self, input):
        return fft.r2c(input)

    def backward(self, grad_output):
        return fft.c2r(grad_output)


def fft_c2c(input):
    return FFT_C2C()(input)

def ifft_c2c(input):
    return FFT_iC2C()(input)

def ifft_c2r(input):
    # return FFT_C2R()(input)
    return FFT_iC2C()(input).select(input.dim()-1, 0)

def fft_r2c(input):
    return FFT_R2C()(input)
