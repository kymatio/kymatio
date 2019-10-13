# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

import tensorflow as tf

from collections import namedtuple

BACKEND_NAME = 'tensorflow'

def modulus_complex(x):
    """Compute the complex modulus.

    Computes the modulus of x and stores the result in a complex tensor of the
    same size, with the real part equal to the modulus and the imaginary part
    equal to zero.

    Parameters
    ----------
    x : tensor
        A complex tensor.

    Returns
    -------
    norm : tensor
        A complex tensor with the same dimensions as x. The real part contains
        the complex modulus.
        of x.
    """

    norm = tf.abs(x)
    return tf.cast(norm, tf.complex64)

def subsample_fourier(x, k):
    """Subsampling in the Fourier domain.

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
    y = tf.reshape(x, (-1, x.shape[1],k,N // k))
    res = tf.reduce_mean(y, axis=(-2))
    return res

def pad_1d(x, pad_left, pad_right, mode='constant', value=0.):
    """Pad real 1D tensors.

    1D implementation of the padding function for real Tensorflow tensors.

    Parameters
    ----------
    x : tensor
        Three-dimensional input tensor with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the tensor (at the beginning of the
        temporal axis).
    pad_right : int
        Amount to add on the right of the tensor (at the end of the temporal
        axis).
    mode : string, optional
        Padding mode. Only option on Tensorflow backend is reflect. 
    value : float, optional
        If mode == 'constant', value to input within the padding. Does nothing
        on Tensorflow backend.

    Returns
    -------
    res : tensor
        The tensor passed along the third dimension.

    """
    if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
        if mode == 'reflect':
            raise ValueError('Indefinite padding size (larger than tensor).')
    paddings = [[0, 0]] * len(x.shape[:-1].as_list())
    paddings += [[pad_left, pad_right]]
    return tf.cast(tf.pad(x, paddings, mode="REFLECT"), tf.complex64)


def pad(x, pad_left=0, pad_right=0, to_complex=True):
    """Pad real 1D tensors and map to complex.

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
        Amount to add on the right of the tensor (at the end of the temporal
        axis).
    to_complex : boolean, optional
        Whether to map the resulting padded tensor to a complex type (seen
        as a real number). Defaults to True.

    Returns
    -------
    output : tensor
        A padded signal, possibly transformed into a four-dimensional tensor. 
    """
    output = pad_1d(x, pad_left, pad_right, mode='reflect')
    return output

def unpad(x, i0, i1):
    """Unpad real 1D tensor.

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

def real(x):
    """Real part of complex tensor.

    Takes the real part of a complex tensor.

    Parameters
    ----------
    x : tensor
        A complex tensor.

    Returns
    -------
    x_real : tensor
        The tensor tf.real(x) which is interpreted as the real part of x.

    """
    return tf.real(x)

def fft1d_c2c(x):
    """Compute the 1D FFT of a complex signal.

    Input
    -----
    x : tensor
        A tensor of size (..., T).

    Returns
    -------
    x_f : tensor
        A tensor of the same size as x containing its Fourier transform in the
        standard Tensorflow FFT ordering.

    """
    return tf.signal.fft(x)

def ifft1d_c2c(x):
    """Compute the normalized 1D inverse FFT of a complex signal.

    Input
    -----
    x_f : tensor
        A tensor of size (..., T). The frequencies are assumed to be in
        the standard Tensorflow FFT ordering.

    Returns
    -------
    x : tensor
        A tensor of the same size of x_f containing the normalized inverse
        Fourier transform of x_f.

    """
    return tf.signal.ifft(x)

def finalize(s0, s1, s2):
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
    return tf.concat([tf.concat(s0, axis=-2), tf.concat(s1, axis=-2), tf.concat(s2, axis=-2)], axis=-2)


backend = namedtuple('backend', ['name', 'modulus_complex', 'subsample_fourier', 'real', 'unpad', 'fft1d_c2c', 'ifft1d_c2c'])
backend.name = 'tensorflow'
backend.modulus_complex = modulus_complex
backend.subsample_fourier = subsample_fourier
backend.real = real
backend.unpad = unpad
backend.pad = pad
backend.pad_1d = pad_1d
backend.fft1d_c2c = fft1d_c2c
backend.ifft1d_c2c = ifft1d_c2c
backend.finalize = finalize
