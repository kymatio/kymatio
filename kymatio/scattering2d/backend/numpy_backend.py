# Authors: Edouard Oyallon, Sergey Zagoruyko

import numpy as np
from collections import namedtuple


BACKEND_NAME = 'numpy'


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
                if set to true, then there is no padding, one simply converts
                from real to complex.

        """
        self.pre_pad = pre_pad
        self.pad_size = pad_size

    def __call__(self, x):
        if self.pre_pad:
            return x
        else:
            np_pad = ((self.pad_size[0], self.pad_size[1]), (self.pad_size[2], self.pad_size[3]))
            batch_shape = x.shape[:-2]
            signal_shape = x.shape[-2:]
            x = x.reshape((-1, 1) + signal_shape)
            output = np.pad(x, ((0,0), (0,0), np_pad[0], np_pad[1]), mode='reflect')
            output = output.reshape(batch_shape + output.shape[-2:])
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
    return np.expand_dims(in_[..., 1:-1, 1:-1], -3)


class SubsampleFourier(object):
    """ Subsampling of a 2D image performed in the Fourier domain.

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
    def __call__(self, x, k):
        batch_shape = x.shape[:-2]
        signal_shape = x.shape[-2:]

        x = x.reshape((-1,) + signal_shape)
        y = x.reshape(-1, k, x.shape[1] // k, k, x.shape[2] // k)

        out = np.zeros_like(y, dtype=x.dtype)
        out = y.mean(axis=(1, 3))

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
        x: input complex tensor.

        Returns
        -------
        output: a real tensor equal to the modulus of x.

    """
    def __call__(self, x):
        norm = np.abs(x)
        return norm


def fft(x, direction='C2C', inverse=False):
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
            NB : if direction is equal to 'C2R', then an error is raised.

    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if direction == 'C2R':
        output = np.real(np.fft.ifft2(x)) * x.shape[-1] * x.shape[-2]
    elif direction == 'C2C':
        if inverse:
            output = np.fft.ifft2(x) * x.shape[-1] * x.shape[-2]
        else:
            output = np.fft.fft2(x)

    return output


def cdgmm(A, B, inplace=False):
    """
        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2)
        B : tensor
            B is a complex tensor of size (M, N) or real tensor of (M, N)
        inplace : boolean, optional
            if set to True, all the operations are performed inplace

        Returns
        -------
        C : tensor
            output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]

    """
    if B.ndim != 2:
        raise RuntimeError('The dimension of the second input must be 2.')

    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B


def empty_like(x, shape):
    return np.empty(shape, x.dtype)


backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'empty_like'])
backend.name = 'numpy'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.empty_like = empty_like
