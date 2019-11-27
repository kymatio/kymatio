# Authors: Edouard Oyallon, Sergey Zagoruyko

import torch
from torch.nn import ReflectionPad2d
from collections import namedtuple

BACKEND_NAME = 'torch'

def _iscomplex(x):
    return x.shape[-1] == 2

def _isreal(x):
    return x.shape[-1] == 1

class Pad(object):
    def __init__(self, pad_size, input_size, pre_pad=False):
        """Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                size of the original signal [height, width].
            pre_pad : boolean, optional
                If set to true, then there is no padding, one simply adds the imaginary part.

        """
        self.pre_pad = pre_pad
        self.pad_size = pad_size
        self.input_size = input_size

        self.build()

    def build(self):
        """Builds the padding module.

            Attributes
            ----------
            padding_module : ReflectionPad2d
                Pads the input tensor using the reflection of the input
                boundary.

        """
        pad_size_tmp = list(self.pad_size)

        # This handles the case where the padding is equal to the image size
        if pad_size_tmp[0] == self.input_size[0]:
            pad_size_tmp[0] -= 1
            pad_size_tmp[1] -= 1
        if pad_size_tmp[2] == self.input_size[1]:
            pad_size_tmp[2] -= 1
            pad_size_tmp[3] -= 1
        # Pytorch expects its padding as [left, right, top, bottom]
        self.padding_module = ReflectionPad2d([pad_size_tmp[2], pad_size_tmp[3],
                                               pad_size_tmp[0], pad_size_tmp[1]])

    def __call__(self, x):
        """Applies padding and maps to complex.

            Parameters
            ----------
            x : tensor
                Real tensor input to be padded and sent to complex domain.

            Returns
            -------
            output : tensor
                Complex torch tensor that has been padded.

        """
        batch_shape = x.shape[:-2]
        signal_shape = x.shape[-2:]
        x = x.reshape((-1, 1) + signal_shape)
        if not self.pre_pad:
            x = self.padding_module(x)
            if self.pad_size[0] == self.input_size[0]:
                x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.shape[2] - 2, :].unsqueeze(2)], 2)
            if self.pad_size[2] == self.input_size[1]:
                x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.shape[3] - 2].unsqueeze(3)], 3)

        output = x.new_zeros(x.shape + (2,))
        output[..., 0] = x
        output = output.reshape(batch_shape + output.shape[-3:])
        return output

def unpad(in_):
    """Unpads input.

        Slices the input tensor at indices between 1:-1.

        Parameters
        ----------
        in_ : tensor
            Input tensor.

        Returns
        -------
        in_[..., 1:-1, 1:-1] : tensor
            Output tensor.  Unpadded input.

    """
    return torch.unsqueeze(in_[..., 1:-1, 1:-1], -3)

class SubsampleFourier(object):
    """Subsampling of a 2D image performed in the Fourier domain

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor
            Input tensor with at least 5 dimensions, the last being the real
            and imaginary parts. Ideally, the last dimension should be a power
            of 2 to avoid errors.
        k : int
            Integer such that x is subsampled by 2**k along the spatial variables.

        Returns
        -------
        out : tensor
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * (2**k), u2 * (2**k)].

    """
    def __call__(self, x, k):
        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')
        batch_shape = x.shape[:-3]
        signal_shape = x.shape[-3:]
        x = x.view((-1,) + signal_shape)
        y = x.view(-1,
                       k, x.shape[1] // k,
                       k, x.shape[2] // k,
                       2)

        out = y.mean(3, keepdim=False).mean(1, keepdim=False)
        out = out.reshape(batch_shape + out.shape[-3:])
        return out

class Modulus(object):
    """This class implements a modulus transform for complex numbers.

        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)

        Parameters
        ---------
        x : tensor
            Complex torch tensor.

        Returns
        -------
        output : tensor
            A tensor with the same dimensions as x, such that output[..., 0]
            contains the complex modulus of x, while output[..., 1] = 0.

    """
    def __call__(self, x):
        norm = torch.zeros_like(x)
        norm[...,0] = (x[...,0]*x[...,0] +
                       x[...,1]*x[...,1]).sqrt()
        return norm

def fft(x, direction='C2C', inverse=False):
    """Interface with torch FFT routines for 2D signals.

        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x)
        x_ifft = fft(x, inverse=True)

        Parameters
        ----------
        x : tensor
            Complex input for the FFT.
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex.
        inverse : bool
            True for computing the inverse FFT.
            NB : If direction is equal to 'C2R', then an error is raised.

        Raises
        ------
        RuntimeError
            In the event that we are going from complex to real and not doing
            the inverse FFT or in the event x is not contiguous.
        TypeError
            In the event that x does not have a final dimension 2 i.e. not
            complex.

        Returns
        -------
        output : tensor
            Result of FFT or IFFT.

    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if not _iscomplex(x):
        raise TypeError('The input should be complex (e.g. last dimension is 2).')

    if not x.is_contiguous():
        raise RuntimeError('Tensors must be contiguous.')

    if direction == 'C2R':
        output = torch.irfft(x, 2, normalized=False, onesided=False) * x.shape[-2] * x.shape[-3]
    elif direction == 'C2C':
        if inverse:
            output = torch.ifft(x, 2, normalized=False) * x.shape[-2] * x.shape[-3]
        else:
            output = torch.fft(x, 2, normalized=False)

    return output

def cdgmm(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2).
        B : tensor
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1).
        inplace : boolean, optional
            If set to True, all the operations are performed in place.

        Raises
        ------
        RuntimeError
            In the event that the filter B is not a 3-tensor with a last
            dimension of size 1 or 2, or A and B are not compatible for
            multiplication.
        TypeError
            In the event that A is not complex, or B does not have a final
            dimension of 1 or 2, or A and B are not of the same dtype, or if
            A and B are not on the same device.

        Returns
        -------
        C : tensor
            Output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

    """
    if not _iscomplex(A):
        raise TypeError('The input must be complex, indicated by a last '
                        'dimension of size 2.')

    if B.ndimension() != 3:
        raise RuntimeError('The filter must be a 3-tensor, with a last '
                           'dimension of size 1 or 2 to indicate it is real '
                           'or complex, respectively.')

    if not _iscomplex(B) and not _isreal(B):
        raise TypeError('The filter must be complex or real, indicated by a '
                        'last dimension of size 2 or 1, respectively.')

    if A.shape[-3:-1] != B.shape[-3:-1]:
        raise RuntimeError('The filters are not compatible for multiplication.')

    if A.dtype is not B.dtype:
        raise TypeError('A and B must be of the same dtype.')

    if B.device.type == 'cuda':
        if A.device.type == 'cuda':
            if A.device.index != B.device.index:
                raise TypeError('A and B must be on the same GPU.')
        else:
            raise TypeError('A must be on GPU.')

    if B.device.type == 'cpu':
        if A.device.type == 'cuda':
            raise TypeError('A must be on CPU.')

    if _isreal(B):
        if inplace:
            return A.mul_(B)
        else:
            return A * B
    else:
        C = A.new(A.shape)

        A_r = A[..., 0].contiguous().view(-1, A.shape[-2]*A.shape[-3])
        A_i = A[..., 1].contiguous().view(-1, A.shape[-2]*A.shape[-3])

        B_r = B[...,0].contiguous().view(B.shape[-2]*B.shape[-3]).unsqueeze(0).expand_as(A_i)
        B_i = B[..., 1].contiguous().view(B.shape[-2]*B.shape[-3]).unsqueeze(0).expand_as(A_r)

        C[..., 0].view(-1, C.shape[-2]*C.shape[-3])[:] = A_r * B_r - A_i * B_i
        C[..., 1].view(-1, C.shape[-2]*C.shape[-3])[:] = A_r * B_i + A_i * B_r

        return C if not inplace else A.copy_(C)

def concatenate(s0, s1, s2):
    """Concatenate scattering of different orders.

        Parameters
        ----------
        s0 : tensor
            Tensor which contains the zeroth order scattering coefficents.
        s1 : tensor
            Tensor which contains the first order scattering coefficents.
        s2 : tensor
            Tensor which contains the second order scattering coefficents.

        Returns
        -------
        s : tensor
            Final output. Scattering transform.

    """
    if len(s2) > 0:
        return torch.cat([torch.cat(s0, -3), torch.cat(s1, -3), torch.cat(s2, -3)], -3)
    else:
        return torch.cat([torch.cat(s0, -3), torch.cat(s1, -3)], -3)

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])
backend.name = 'torch'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.concatenate = concatenate
