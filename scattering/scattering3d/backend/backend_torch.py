from collections import defaultdict
from skcuda import cufft
import torch
import numpy as np


try:
    import pyfftw
    FFTW = True
except:
    import scipy.fftpack as fft
    FFTW = False


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


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



class Fft3d(object):
    """This class builds a wrapper to 3D FFTW on CPU / cuFFT on nvidia GPU."""

    def __init__(self, n_fftw_threads=8):
        self.n_fftw_threads = n_fftw_threads
        self.fftw_cache = defaultdict(lambda: None)
        self.cufft_cache = defaultdict(lambda: None)

    def buildCufftCache(self, input, type):
        batch_size, M, N, O, _ = input.size()
        signal_dims = np.asarray([M, N, O], np.int32)
        batch = batch_size
        idist = M * N * O
        istride = 1
        ostride = istride
        odist = idist
        rank = 3
        plan = cufft.cufftPlanMany(rank, signal_dims.ctypes.data,
                                         signal_dims.ctypes.data,
                                   istride, idist, signal_dims.ctypes.data, 
                                   ostride, odist, type, batch)
        self.cufft_cache[(input.size(), type, input.get_device())] = plan

    def buildFftwCache(self, input, inverse):
        direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'
        batch_size, M, N, O, _ = input.size()
        fftw_input_array = pyfftw.empty_aligned(
            (batch_size, M, N, O), dtype='complex64')
        fftw_output_array = pyfftw.empty_aligned(
            (batch_size, M, N, O), dtype='complex64')
        fftw_object = pyfftw.FFTW(fftw_input_array, fftw_output_array,
                                  axes=(1, 2, 3), direction=direction,
                                  threads=self.n_fftw_threads)
        self.fftw_cache[(input.size(), inverse)] = (
            fftw_input_array, fftw_output_array, fftw_object)

    def __call__(self, input, inverse=False, normalized=False):
        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:

                if FFTW:
                    # XXX the below code seems weird. Complex numbers are already
                    #       organized that way and this creates a copy. fix pls
                    def f(x): return np.stack([x.real, x.imag], axis=len(x.shape))
                    if(self.fftw_cache[(input.size(), inverse)] is None):
                        self.buildFftwCache(input, inverse)
                    input_arr, output_arr, fftw_obj = self.fftw_cache[(
                        input.size(), inverse)]

                    input_arr.real[:] = input[..., 0]
                    input_arr.imag[:] = input[..., 1]
                    fftw_obj()

                    return torch.from_numpy(f(output_arr))
                else:
                    # XXX there might be a normalization factor off here
                    #       but this here is about getting rid of the
                    #       hard pyfftw dependency. It should be fixed
                    #       as soon as we have tests for this whole code block
                    if inverse:
                        return fft.ifftn(input, axes=(-1, -2, -3))
                    else:
                        return fft.fftn(input, axes=(-1, -2, -3))

        if not input.is_contiguous():
            raise(RuntimeError("input is not contiguous"))

        output = input.new(input.size())
        flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
        ffttype = cufft.CUFFT_C2C if isinstance(
            input, torch.cuda.FloatTensor) else cufft.CUFFT_Z2Z
        if self.cufft_cache[
                    (input.size(), ffttype, input.get_device())] is None:
            self.buildCufftCache(input, ffttype)
        cufft.cufftExecC2C(self.cufft_cache[(input.size(), ffttype, 
                    input.get_device())], input.data_ptr(),
                    output.data_ptr(), flag)
        if normalized:
            output /= input.size(1) * input.size(2) * input.size(3)
        return output


def cdgmm3d(A, B):
    """
    Pointwise multiplication of complex tensors.

    ----------
    A: complex torch tensor
    B: complex torch tensor of the same size as A

    Returns
    -------
    output : torch tensor of the same size as A containing the result of the 
             elementwise complex multiplication of  A with B 
    """
    if not A.is_contiguous():
        warnings.warn("cdgmm3d: tensor A is converted to a contiguous array")
        A = A.contiguous()
    if not B.is_contiguous():
        warnings.warn("cdgmm3d: tensor B is converted to a contiguous array")
        B = B.contiguous()

    if A.size()[-4:] != B.size():
        raise RuntimeError(
            'The tensors are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 4:
        raise RuntimeError('The second tensor must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    C = torch.empty_like(A)

    C[..., 0] = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    C[..., 1] = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]

    return C
