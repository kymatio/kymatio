import tensorflow as tf
from collections import namedtuple

BACKEND_NAME = 'tensorflow'


#from ...backend.tensorflow_backend import Modulus, cdgmm, concatenate, complex_check, real_check

from ...backend.tensorflow_backend import TensorFlowBackend

class Pad(object):
    def __init__(self, pad_size, input_size):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : list of 4 integers
                size of padding to apply.
            input_size : list of 2 integers
                size of the original signal
        """
        self.pad_size = pad_size

    def __call__(self, x):
        paddings = [[0, 0]] * len(x.shape[:-2])
        paddings += [[self.pad_size[0], self.pad_size[1]], [self.pad_size[2], self.pad_size[3]]]
        return tf.pad(x, paddings, mode="REFLECT")

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
    """ Subsampling of a 2D image performed in the Fourier domain.

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least three dimensions.
        k : int
            integer such that x is subsampled by k along the spatial variables.

        Returns
        -------
        out : tensor_like
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k]

    """
    def __call__(self, x, k):
        complex_check(x)

        y = tf.reshape(x, (-1, k, x.shape[1] // k, k, x.shape[2] // k))

        out = tf.reduce_mean(y, axis=(1, 3))
        return out


def rfft(x):
    real_check(x)
    return tf.signal.fft2d(tf.cast(x, tf.complex64), name='rfft2d')


def irfft(x):
    complex_check(x)
    return tf.math.real(tf.signal.ifft2d(x, name='irfft2d'))


def ifft(x):
    complex_check(x)
    return tf.signal.ifft2d(x, name='ifft2d')


#backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])

backend = TensorFlowBackend()

#backend.name = 'tensorflow'
#backend.cdgmm = cdgmm
#backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.rfft = rfft
backend.irfft = irfft
backend.ifft = ifft
backend.Pad = Pad
backend.unpad = unpad
#backend.concatenate = lambda x: concatenate(x, -3)

real_check = backend.real_check
complex_check = backend.complex_check

