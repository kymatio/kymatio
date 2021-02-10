# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux
from collections import namedtuple
import scipy
import numpy as np

BACKEND_NAME = 'numpy'

from ...backend.numpy_backend import modulus, cdgmm, real
from ...backend.base_backend import FFT


def subsample_fourier(x, k):
    """Subsampling in the Fourier domain
    Subsampling in the temporal domain amounts to periodization in the Fourier
    domain, so the input is periodized according to the subsampling factor.
    Parameters
    ----------
    x : tensor
        Input tensor with at least 3 dimensions, where the next to last
        corresponds to the frequency index in the standard PyTorch FFT
        ordering. The length of this dimension should be a power of 2 to
        avoid errors. The last dimension should represent the real and
        imaginary parts of the Fourier transform.
    k : int
        The subsampling factor.
    Returns
    -------
    res : tensor
        The input tensor periodized along the next to last axis to yield a
        tensor of size x.shape[-2] // k along that dimension.
    """

    N = x.shape[-1]
    res = x.reshape(x.shape[:-1] + (k, N // k)).mean(axis=(-2,))
    return res


def pad_1d(x, pad_left, pad_right, mode='constant', value=0.):
    """Pad real 1D tensors
    1D implementation of the padding function for real PyTorch tensors.
    Parameters
    ----------
    x : tensor
        Three-dimensional input tensor with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the tensor (at the beginning of the
        temporal axis).
    pad_right : int
        amount to add on the right of the tensor (at the end of the temporal
        axis).
    mode : string, optional
        Padding mode. Options include 'constant' and 'reflect'. See the
        PyTorch API for other options.  Defaults to 'constant'.
    value : float, optional
        If mode == 'constant', value to input within the padding. Defaults to
        0.
    Returns
    -------
    res : tensor
        The tensor passed along the third dimension.
    """
    if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
        if mode == 'reflect':
            raise ValueError('Indefinite padding size (larger than tensor).')

    res = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right),), mode=mode)
    return res


def pad(x, pad_left=0, pad_right=0):
    """Pad real 1D tensors and map to complex
    Padding which allows to simultaneously pad in a reflection fashion and map
    to complex if necessary.
    Parameters
    ----------
    x : tensor
        Three-dimensional input tensor with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the tensor (at the beginning of the
        temporal axis).
    pad_right : int
        amount to add on the right of the tensor (at the end of the temporal
        axis).
    Returns
    -------
    output : tensor
        A padded signal.
    """
    output = pad_1d(x, pad_left, pad_right, mode='reflect')
    return output


def unpad(x, i0, i1):
    """Unpad real 1D tensor
    Slices the input tensor at indices between i0 and i1 along the last axis.
    Parameters
    ----------
    x : tensor
        Input tensor with least one axis.
    i0 : int
        Start of original signal before padding.
    i1 : int
        End of original signal before padding.
    Returns
    -------
    x_unpadded : tensor
        The tensor x[..., i0:i1].
    """
    return x[..., i0:i1]




def concatenate(arrays):
    return np.stack(arrays, axis=-2)


backend = namedtuple('backend',
                     ['name', 'modulus_complex', 'subsample_fourier', 'real',
                      'unpad', 'fft', 'concatenate', 'cdgmm'])
backend.name = 'numpy'
backend.modulus_complex = modulus
backend.subsample_fourier = subsample_fourier
backend.real = real
backend.unpad = unpad
backend.pad = pad
backend.pad_1d = pad_1d
backend.cdgmm = cdgmm
backend.fft = FFT(lambda x:scipy.fftpack.fft(x),
                  lambda x:scipy.fftpack.ifft(x),
                  lambda x:np.real(scipy.fftpack.ifft(x)),
                  lambda x:None)
backend.concatenate = concatenate
