import torch
import warnings
from skcuda import cublas

BACKEND_NAME = 'torch_skcuda'

from collections import namedtuple

def iscomplex(input):
    return input.size(-1) == 2


def to_complex(input):
    output = input.new(input.size() + (2,)).fill_(0)
    output[..., 0] = input
    return output


def complex_modulus(input_array):
    modulus = torch.zeros_like(input_array)
    modulus[..., 0] += torch.sqrt((input_array ** 2).sum(-1))
    return modulus


def fft(input, inverse=False):
    """
        fft of a 3d signal

        Example
        -------

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        inverse : bool
            True for computing the inverse FFT.
    """
    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))
    if inverse:
        return torch.ifft(input, 3)
    return torch.fft(input, 3)


def cdgmm3d(A, B, inplace=False):
    """
    Pointwise multiplication of complex tensors.

    ----------
    A: complex tensor
    B: complex tensor of the same size as A

    Returns
    -------
    output : tensor of the same size as A containing the result of the
             elementwise complex multiplication of  A with B
    """
    if not A.is_contiguous():
        warnings.warn("cdgmm3d: tensor A is converted to a contiguous array")
        A = A.contiguous()
    if not B.is_contiguous():
        warnings.warn("cdgmm3d: tensor B is converted to a contiguous array")
        B = B.contiguous()

    if A.size()[-4:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication.')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex.')

    if B.ndimension() != 4:
        raise RuntimeError('The filters must be simply a complex array.')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type.')

    if not A.is_cuda:
        raise RuntimeError('Use the torch backend for cpu tensors.')

    C = A.new(A.size()) if not inplace else A
    m, n = B.nelement() // 2, A.nelement() // B.nelement()
    lda = m
    ldc = m
    incx = 1
    handle = torch.cuda.current_blas_handle()
    stream = torch.cuda.current_stream()._as_parameter_
    cublas.cublasSetStream(handle, stream)
    cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
    return C


from .torch_backend import finalize
from .torch_backend import modulus_rotation
from .torch_backend import subsample
from .torch_backend import compute_integrals
from .torch_backend import _compute_local_scattering_coefs
from .torch_backend import _compute_standard_scattering_coefs
from .torch_backend import aggregate

backend = namedtuple('backend', ['name', 'cdgmm3d', 'fft', 'finalize', 'modulus', 'modulus_rotation', 'subsample', \
                                     'compute_integrals', 'to_complex', 'aggregate'])


backend.name = 'skcuda'
backend.aggregate = aggregate
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.to_complex = to_complex
backend.finalize = finalize
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.subsample = subsample
backend.compute_integrals = compute_integrals
backend._compute_standard_scattering_coefs = _compute_standard_scattering_coefs
backend._compute_local_scattering_coefs = _compute_local_scattering_coefs

