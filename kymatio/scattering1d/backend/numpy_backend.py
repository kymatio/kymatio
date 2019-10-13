# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

import numpy as np

from collections import namedtuple

BACKEND_NAME = 'numpy'

def modulus_complex(x):
    """Compute the complex modulus.

    Computes the modulus of x and stores the result in a real numpy array.

    Parameters
    ----------
    x : numpy array
        A complex numpy array.

    Returns
    -------
    norm : numpy array
        A real numpy array with the same dimensions as x and is the complex
        modulus of x.

    """

    norm = np.abs(x)

    return norm

def subsample_fourier(x, k):
    """Subsampling in the Fourier domain.

    Subsampling in the temporal domain amounts to periodization in the Fourier
    domain, so the input is periodized according to the subsampling factor.

    Parameters
    ----------
    x : numpy array
        Input numpy array with at least 3 dimensions.
    k : int
        The subsampling factor.

    Returns
    -------
    res : numpy array
        The input numpy array periodized along the next to last axis to yield a
        numpy array of size x.shape[-2] // k along that dimension.

    """

    N = x.shape[-1]
    res = x.reshape(x.shape[:-2] + (k, N // k)).mean(axis=-2, keepdims=True)
    return res

def pad_1d(x, pad_left, pad_right, mode='constant', value=0.):
    """Pad real 1D numpy arrays.

    1D implementation of the padding function for real numpy arrays.

    Parameters
    ----------
    x : numpy array
        Three-dimensional input numpy array with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the array (at the beginning of the
        temporal axis).
    pad_right : int
        Amount to add on the right of the array (at the end of the temporal
        axis).
    mode : string, optional
        Padding mode. Options for numpy backend only includes 'reflect'.
    value : float, optional
        Does nothing on numpy backend.

    Returns
    -------
    res : numpy array
        The numpy array passed along the third dimension.
    """
    if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
        if mode == 'reflect':
            raise ValueError('Indefinite padding size (larger than numpy array).')
    return np.pad(x, ((0,0), (0,0), (pad_left, pad_right)), mode='reflect')

def pad(x, pad_left=0, pad_right=0, to_complex=True):
    """Pad 1D numpy arrays.

    Padding which allows to simultaneously pad in a reflection fashion and map
    to complex if necessary.

    Parameters
    ----------
    x : numpy array
        Three-dimensional input numpy array with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the numpy array (at the beginning of the
        temporal axis).
    pad_right : int
        Amount to add on the right of the numpy array (at the end of the temporal
        axis).
    to_complex : boolean, optional
        Whether to map the resulting padded numpy array to a complex type (seen
        as a real number). Defaults to True. Does nothing on numpy backend.

    Returns
    -------
    output : numpy array
        A padded signal, possibly transformed into a four-dimensional numpy
        array.

    """
    output = pad_1d(x, pad_left, pad_right, mode='reflect')
    return output

def unpad(x, i0, i1):
    """Unpad real 1D numpy array.

    Slices the input numpy array at indices between i0 and i1 along the last axis.

    Parameters
    ----------
    x : numpy array
        Input numpy array with least one axis.
    i0 : int
        Start of original signal before padding.
    i1 : int
        End of original signal before padding.

    Returns
    -------
    x_unpadded : numpy array
        The numpy array x[..., i0:i1].

    """
    return x[..., i0:i1]

def real(x):
    """Real part of complex numpy array.

    Takes the real part of a complex numpy array.

    Parameters
    ----------
    x : numpy array
        A complex numpy array.

    Returns
    -------
    x_real : numpy array
        The numpy array np.real(x) which is interpreted as the real part of x.

    """
    return np.real(x)

def fft1d_c2c(x):
    """Compute the 1D FFT of a complex signal.

    Input
    -----
    x : numpy array
        A numpy array of size (..., T).

    Returns
    -------
    x_f : numpy array
        A numpy array of the same size as x containing its Fourier transform in the
        standard numpy FFT ordering.

    """
    return np.fft.fft(x)

def ifft1d_c2c(x):
    """Compute the normalized 1D inverse FFT of a complex signal.

    Input
    -----
    x_f : numpy array
        A numpy array of size (..., T). The frequencies are assumed to be in
        the standard Numpy FFT ordering.

    Returns
    -------
    x : numpy array
        A numpy array of the same size of x_f containing the normalized inverse
        Fourier transform of x_f.

    """
    return np.fft.ifft(x)

def finalize(s0, s1, s2):
    """Concatenate scattering of different orders.

    Parameters
    ----------
    s0 : numpy array
        Numpy array which contains the zeroth order scattering coefficents.
    s1 : numpy array
        Numpy array which contains the first order scattering coefficents.
    s2 : numpy array
        Numpy array which contains the second order scattering coefficents.
    
    Returns
    -------
    s : numpy array
        Final output. Scattering transform.

    """
    return np.concatenate([np.concatenate(s0, axis=-2), np.concatenate(s1, axis=-2), np.concatenate(s2, axis=-2)], axis=-2)


backend = namedtuple('backend', ['name', 'modulus_complex', 'subsample_fourier', 'real', 'unpad', 'fft1d_c2c', 'ifft1d_c2c'])
backend.name = 'torch'
backend.modulus_complex = modulus_complex
backend.subsample_fourier = subsample_fourier
backend.real = real
backend.unpad = unpad
backend.pad = pad
backend.pad_1d = pad_1d
backend.fft1d_c2c = fft1d_c2c
backend.ifft1d_c2c = ifft1d_c2c
backend.finalize = finalize
