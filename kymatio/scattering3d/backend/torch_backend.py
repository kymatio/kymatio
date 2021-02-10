import torch
import torch.fft
import warnings

BACKEND_NAME = 'torch'
from collections import namedtuple

from ...backend.torch_backend import cdgmm, Modulus, complex_check, real_check

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
        module = x.abs()**2
    else:
        module = module ** 2 + x.abs()**2
    return torch.sqrt(module)


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
    integrals = torch.zeros((input_array.shape[0], len(integral_powers)), device=input_array.device)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = (input_array ** q).view(
            input_array.shape[0], -1).sum(1)
    return integrals


def concatenate(arrays, L):
    S = torch.stack(arrays, dim=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    return S


# we cast to complex here then fft rather than use torch.rfft.
def rfft(x):
    real_check(x)
    x_r = x + 1j*0
    return torch.fft.fftn(x_r, dim=(-1,-2,-3))


def ifft(x):
    complex_check(x)
    return torch.fft.ifftn(x, dim=(-1,-2,-3))



backend = namedtuple('backend',
                     ['name',
                      'cdgmm3d',
                      'fft',
                      'modulus',
                      'modulus_rotation',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'torch'
backend.cdgmm3d = cdgmm
backend.rfft = rfft
backend.ifft = ifft
backend.concatenate = concatenate
backend.modulus = Modulus()
backend.modulus_rotation = modulus_rotation
backend.compute_integrals = compute_integrals
