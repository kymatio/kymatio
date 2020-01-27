import torch
import warnings
from skcuda import cublas

BACKEND_NAME = 'torch_skcuda'

from collections import namedtuple


def _is_complex(input):
    return input.shape[-1] == 2


def cdgmm3d(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : torch tensor
            Complex torch tensor.
        B : torch tensor
            Complex of the same size as A.
        inplace : boolean, optional
            If set True, all the operations are performed inplace.

        Raises
        ------
        RuntimeError
            In the event that the tensors are not compatibile for multiplication
            (i.e. the final four dimensions of A do not match with the dimensions
            of B), or in the event that B is not complex, or in the event that the
            type of A and B are not the same.
        TypeError
            In the event that x is not complex i.e. does not have a final dimension
            of 2, or in the event that both tensors are not on the same device.

        Returns
        -------
        output : torch tensor
            Torch tensor of the same size as A containing the result of the
            elementwise complex multiplication of A with B.

    """
    if not A.is_contiguous():
        warnings.warn("cdgmm3d: tensor A is converted to a contiguous array")
        A = A.contiguous()
    if not B.is_contiguous():
        warnings.warn("cdgmm3d: tensor B is converted to a contiguous array")
        B = B.contiguous()

    if A.shape[-4:] != B.shape:
        raise RuntimeError('The filters are not compatible for multiplication.')

    if not _is_complex(A) or not _is_complex(B):
        raise TypeError('The input, filter and output should be complex.')

    if B.ndimension() != 4:
        raise RuntimeError('The filters must be simply a complex array.')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type.')

    if not A.is_cuda:
        raise RuntimeError('Use the torch backend for CPU tensors.')

    C = A.new(A.shape) if not inplace else A
    m, n = B.nelement() // 2, A.nelement() // B.nelement()
    lda = m
    ldc = m
    incx = 1
    handle = torch.cuda.current_blas_handle()
    stream = torch.cuda.current_stream()._as_parameter_
    cublas.cublasSetStream(handle, stream)
    cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
    return C


from .torch_backend import complex_modulus
from .torch_backend import fft
from .torch_backend import modulus_rotation
from .torch_backend import compute_integrals
from .torch_backend import concatenate

backend = namedtuple('backend', ['name', 'cdgmm3d', 'fft', 'modulus', 'modulus_rotation',
                                 'compute_integrals', 'concatenate'])

backend.name = 'torch_skcuda'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.concatenate = concatenate
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.compute_integrals = compute_integrals
