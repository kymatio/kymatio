# Authors: Edouard Oyallon, Sergey Zagoruyko

import tensorflow as tf
from collections import namedtuple

BACKEND_NAME = 'tensorflow'

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

    def __call__(self, x):
        if self.pre_pad:
            return x
        else:
            paddings = [[0, 0]]*len(x.shape[:-2].as_list())
            paddings += [[self.pad_size[0],self.pad_size[1]],[self.pad_size[2],self.pad_size[3]]]
            return tf.cast(tf.pad(x, paddings, mode="REFLECT"), tf.complex64)

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
    return tf.expand_dims(in_[..., 1:-1, 1:-1],-3)

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
    def __call__(self, x, k):
        y = tf.reshape(x,(-1,x.shape[1],
                       x.shape[2]//(x.shape[2] // k), x.shape[2] // k,
                       x.shape[3]//(x.shape[3] // k), x.shape[3] // k))

        out = tf.reduce_mean(y, axis=(2, 4))
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
        norm = tf.abs(x)
        return tf.cast(norm, tf.complex64)


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
        output = tf.real(tf.signal.ifft2d(x, name='irfft2d'))*tf.cast(x.shape[-1]*x.shape[-2],tf.float32)
    elif direction == 'C2C':
        if inverse:
            output = tf.signal.ifft2d(x, name='ifft2d')*tf.cast(x.shape[-1]*x.shape[-2], tf.complex64)
        else:
            output = tf.signal.fft2d(x, name='fft2d')

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
    return A * B

def finalize(s0, s1, s2):
    """ Concatenate scattering of different orders"""
    return tf.concat([tf.concat(s0, axis=-3), tf.concat(s1, axis=-3), tf.concat(s2, axis=-3)], axis=-3)

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'finalize'])
backend.name = 'numpy'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.finalize = finalize