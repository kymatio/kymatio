# Authors: Edouard Oyallon, Sergey Zagoruyko

import numpy as np
from collections import namedtuple

BACKEND_NAME = 'numpy'

class Pad(object):
    def __init__(self, pad_size, input_size, pre_pad=False):
        """Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                Size of the original signal [height, width].
            pre_pad : boolean, optional
                If set to true, then there is no padding, one simply adds the imaginary part.

            Attributes
            ----------
            pad_size : list of 4 integers 
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                Size of the original signal [height, width].
            pre_pad : boolean
                If set to true, then there is no padding, one simply adds the imaginary part.

        """
        self.pre_pad = pre_pad
        self.pad_size = pad_size

    def __call__(self, x):
        """Applies padding.
            
            Parameters
            ----------
            x : numpy array
                Input to be padded.

            Returns
            -------
            output : numpy array
                Numpy array that has been padded.

        """
        if self.pre_pad:
            return x
        else:
            np_pad = ((self.pad_size[0], self.pad_size[1]), (self.pad_size[2], self.pad_size[3]))
            return np.pad(x, ((0,0), (0,0), np_pad[0], np_pad[1]), mode='reflect')

def unpad(in_):
    """Unpads input.

        Slices the input numpy array at indices between 1::-1.
        
        Parameters
        ----------
        in_ : numpy array_like
            Input numpy array.
        
        Returns
        -------
        in_[..., 1:-1, 1:-1] : numpy array_like
            Output numpy array. Unpadded input.

    """
    return np.expand_dims(in_[..., 1:-1, 1:-1],-3)

class SubsampleFourier(object):
    """Subsampling of a 2D image performed in the Fourier domain.
        
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.
        
        Parameters
        ----------
        x : numpy array_like
            Input array with at least 4 dimensions.
        k : int
            Integer such that x is subsampled by 2**k along the spatial variables.
        
        Returns
        -------
        out : numpy array_like
            Numpy array such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            FFT^{-1}(res)[u1, u2] = FFT^{-1}(x)[u1 * (2**k), u2 * (2**k)].

    """
    def __call__(self, x, k):
        out = np.zeros((x.shape[0], x.shape[1], x.shape[2] // k, x.shape[3] // k), dtype=x.dtype)

        y = x.reshape((x.shape[0], x.shape[1],
                       x.shape[2]//out.shape[2], out.shape[2],
                       x.shape[3]//out.shape[3], out.shape[3]))

        out = y.mean(axis=(2, 4))
        return out

class Modulus(object):
    """This class implements a modulus transform for complex numbers.
        
        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)
        
        Parameters
        ---------
        x : numpy array
            Input complex numpy array.
        
        Returns
        -------
        output : numpy array
            A real numpy array equal to the modulus of x.

    """
    def __call__(self, x):
        norm = np.abs(x)
        return norm

def fft(x, direction='C2C', inverse=False):
    """Interface with numpy FFT routines for 2D signals.
        
        Example
        -------
        x = numpy.random.randn(128, 32, 32, 2).view(numpy.complex64)
        x_fft = fft(x)
        x_ifft = fft(x, inverse=True)
        
        Parameters
        ----------
        input : numpy array
            Complex input for the FFT.
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex.
        inverse : bool
            True for computing the inverse FFT.
            NB : If direction is equal to 'C2R', then an error is raised.

        Raises
        ------
        RuntimeError
            Raised in event we attempt to map from complex to real without
            inverse FFT.

        Returns
        -------
        output : numpy array
            Numpy array of the same size as A containing the result of the
            elementwise complex multiplication of A with B.
            
    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if direction == 'C2R':
        output = np.real(np.fft.ifft2(x))*x.shape[-1]*x.shape[-2]
    elif direction == 'C2C':
        if inverse:
            output = np.fft.ifft2(x)*x.shape[-1]*x.shape[-2]
        else:
            output = np.fft.fft2(x)

    return output

def cdgmm(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) numpy array A and
        numpy array B.

        Parameters
        ----------
        A : numpy array
            A is a complex numpy array of size (B, C, M, N).
        B : numpy array
            B is a complex numpy array of size (M, N) or real numpy array of
            (M, N).
        inplace : boolean, optional
            If set to True, all the operations are performed inplace.

        Raises
        ------
        RuntimeError
            Raised in event B is not two dimensional.
        
        Returns
        -------
        C : numpy array
            Output numpy array of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

    """
    if B.ndim != 2:
        raise RuntimeError('The dimension of the second input must be 2.')

    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B

def finalize(s0, s1, s2):
    """Concatenate scattering of different orders.

    Parameters
    ----------
    s0 : numpy array
        numpy array which contains the zeroth order scattering coefficents.
    s1 : numpy array
        numpy array which contains the first order scattering coefficents.
    s2 : numpy array
        numpy array which contains the second order scattering coefficents.
    
    Returns
    -------
    s : numpy array
        Final output. Scattering transform.

    """
    return np.concatenate([np.concatenate(s0, axis=-3), np.concatenate(s1, axis=-3), np.concatenate(s2, axis=-3)], axis=-3)

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'finalize'])
backend.name = 'numpy'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.finalize = finalize
