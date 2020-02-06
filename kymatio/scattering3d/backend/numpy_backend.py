import numpy as np
from collections import namedtuple
import scipy.fft

BACKEND_NAME = 'numpy'

from ...backend.numpy_backend import modulus, cdgmm
from ...backend.base_backend import FFT


def modulus_rotation(x, module=None):
    """Used for computing rotation invariant scattering transform coefficents.

        Parameters
        ----------
        x : tensor
            Size (batchsize, M, N, O, 2).
        module : tensor
            Tensor that holds the overall sum. If none, initializes the tensor
            to zero (default).
        Returns
        -------
        output : torch tensor
            Tensor of the same size as input_array. It holds the output of
            the operation::
            $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$
            which is covariant to 3D translations and rotations.
    """
    if module is None:
        module = np.zeros_like(x)
    else:
        module = module ** 2
    module += np.abs(x) ** 2
    return np.sqrt(module)


def compute_integrals(input_array, integral_powers):
    """Computes integrals.

        Computes integrals of the input_array to the given powers.
        Parameters
        ----------
        input_array : torch tensor
            Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
            dims.
        integral_powers : list
            List of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p
            norms).
        Returns
        -------
        integrals : torch tensor
            Tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms).
    """
    integrals = np.zeros((input_array.shape[0], len(integral_powers)),
            dtype=np.complex64)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = (input_array ** q).reshape((input_array.shape[0], -1)).sum(axis=1)
    return integrals


def concatenate(arrays, L):
    S = np.stack(arrays, axis=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    return S


backend = namedtuple('backend',
                     ['name',
                      'cdgmm3d',
                      'fft',
                      'modulus',
                      'modulus_rotation',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'numpy'
backend.cdgmm3d = cdgmm
backend.fft = FFT(lambda x:scipy.fft.fftn(x),
                  lambda x:scipy.fft.fftn(x),
                  lambda x:scipy.fft.ifftn(x),
                  lambda x:np.real(scipy.fft.ifftn(x)),
                  lambda x:None)
backend.concatenate = concatenate
backend.modulus = modulus
backend.modulus_rotation = modulus_rotation
backend.compute_integrals = compute_integrals
