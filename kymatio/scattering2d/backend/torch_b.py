# Authors: Edouard Oyallon, Sergey Zagoruyko

import torch
from torch.nn import ReflectionPad2d
import numpy as np

NAME = 'torch'


def iscomplex(input):
    return input.size(-1) == 2


def isreal(input):
    return input.size(-1) == 1


class Pad(object):
    def __init__(self, pad_size, input_size, pre_pad=False):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : list of 4 integers
                size of padding to apply.
            input_size : list of 2 integers
                size of the original signal
            pre_pad : boolean
                if set to true, then there is no padding, one simply adds the imaginarty part.
        """
        self.pre_pad = pre_pad
        self.pad_size = pad_size
        self.input_size = input_size

        self.build()

    def build(self):
        pad_size_tmp = self.pad_size

        # This allow to handle the case where the padding is equal to the image size
        if pad_size_tmp[0] == self.input_size[0]:
            pad_size_tmp[0] -= 1
            pad_size_tmp[1] -= 1
        if pad_size_tmp[2] == self.input_size[1]:
            pad_size_tmp[2] -= 1
            pad_size_tmp[3] -= 1
        self.padding_module = ReflectionPad2d(pad_size_tmp)

    def __call__(self, x):
        if not self.pre_pad:
            x = self.padding_module(x)
            if self.pad_size[0] == self.input_size[0]:
                x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.size(2) - 2, :].unsqueeze(2)], 2)
            if self.pad_size[2] == self.input_size[1]:
                x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.size(3) - 2].unsqueeze(3)], 3)

        output = x.new_zeros(x.shape + (2,))
        output[...,0] = x
        return output

def unpad(in_):
    """
        Slices the input tensor at indices between 1::-1

        Parameters
        ----------
        in_ : tensor_like
            input tensor

        Returns
        -------
        in_[..., 1:-1, 1:-1]
    """
    return in_[..., 1:-1, 1:-1]

class SubsampleFourier(object):
    """
        Subsampling of a 2D image performed in the Fourier domain
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least 5 dimensions, the last being the real
             and imaginary parts.
            Ideally, the last dimension should be a power of 2 to avoid errors.
        k : int
            integer such that x is subsampled by 2**k along the spatial variables.

        Returns
        -------
        res : tensor_like
            tensor such that its fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            FFT^{-1}(res)[u1, u2] = FFT^{-1}(x)[u1 * (2**k), u2 * (2**k)]
    """
    def __call__(self, input, k):
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)


        y = input.view(input.size(0), input.size(1),
                       input.size(2)//out.size(2), out.size(2),
                       input.size(3)//out.size(3), out.size(3),
                       2)

        out = y.mean(4, keepdim=False).mean(2, keepdim=False)
        return out


class Modulus(object):
    """
        This class implements a modulus transform for complex numbers.

        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)

        Parameters
        ---------
        x: input tensor, with last dimension = 2 for complex numbers

        Returns
        -------
        output: a tensor with imaginary part set to 0, real part set equal to
        the modulus of x.
    """
    def __call__(self, x):

        norm = torch.zeros_like(x)
        norm[...,0] = (x[...,0]*x[...,0] +
                       x[...,1]*x[...,1]).sqrt()
        return norm




def fft(input, direction='C2C', inverse=False):
    """
        Interface with torch FFT routines for 2D signals.

        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then the transform
            is automatically inverse.
    """
    if direction == 'C2R':
        inverse = True

    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

    if (not input.is_contiguous()):
        raise (RuntimeError('Tensors must be contiguous!'))

    if direction == 'C2R':
        output = torch.irfft(input, 2, normalized=False, onesided=False) * input.size(-2) * input.size(-3)
    elif direction == 'C2C':
        if inverse:
            output = torch.ifft(input, 2, normalized=False) * input.size(-2) * input.size(-3)
        else:
            output = torch.fft(input, 2, normalized=False)

    return output




def cdgmm(A, B, inplace=False):
    """
        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2)
        B : tensor
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1)
        inplace : boolean, optional
            if set to True, all the operations are performed inplace

        Returns
        -------
        C : tensor
            output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]
    """
    if not iscomplex(A):
        raise TypeError('The input must be complex, indicated by a last '
                        'dimension of size 2')

    if B.ndimension() != 3:
        raise RuntimeError('The filter must be a 3-tensor, with a last '
                           'dimension of size 1 or 2 to indicate it is real '
                           'or complex, respectively')

    if not iscomplex(B) and not isreal(B):
        raise TypeError('The filter must be complex or real, indicated by a '
                        'last dimension of size 2 or 1, respectively')

    if A.size()[-3:-1] != B.size()[-3:-1]:
        raise RuntimeError('The filters are not compatible for multiplication!')

    if A.dtype is not B.dtype:
        raise RuntimeError('A and B must be of the same dtype')

    if A.device.type != B.device.type:
        raise RuntimeError('A and B must be of the same device type')

    if A.device.type == 'cuda':
        if A.device.index != B.device.index:
            raise RuntimeError('A and B must be on the same GPU!')

    if isreal(B):
        if inplace:
            return A.mul_(B)
        else:
            return A * B
    else:
        C = A.new(A.size())

        A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
        A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

        B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
        B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

        C[..., 0].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_r - A_i * B_i
        C[..., 1].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_i + A_i * B_r

        return C if not inplace else A.copy_(C)
