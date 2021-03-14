# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

import torch

BACKEND_NAME = 'torch17'

from .torch_backend import backend


def fft(input, inverse=False):
    """Interface with torch FFT routines for 3D signals.
        fft of a 3d signal
        Example
        -------
        x = torch.randn(128, 32, 32, 32, 2)
        x_fft = fft(x)
        x_ifft = fft(x, inverse=True)
        Parameters
        ----------
        x : tensor
            Complex input for the FFT.
        inverse : bool
            True for computing the inverse FFT.
        Raises
        ------
        TypeError
            In the event that x does not have a final dimension 2 i.e. not
            complex.
        Returns
        -------
        output : tensor
            Result of FFT or IFFT.
    """
    if not _is_complex(input):
        raise TypeError('The input should be complex (e.g. last dimension is 2)')
    if inverse:
        return torch.ifft(input, 3)
    return torch.fft(input, 3)


backend.fft = fft
backend.name = 'torch17'
