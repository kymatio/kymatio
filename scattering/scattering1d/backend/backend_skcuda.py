# Authors: Edouard Oyallon, Joakim Anden

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
import cupy
from collections import namedtuple
from string import Template

NAME = 'skcuda'

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

Stream = namedtuple('Stream', ['ptr'])

def get_dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

def is_complex(input):
    return input.size(-1) == 2

class Modulus(object):
    """Stable complex modulus

    This class implements a modulus transform for complex numbers which is
    stable with respect to very small inputs (z close to 0), avoiding
    returning nans in all cases.

    Usage
    -----
    modulus = ModulusStable.apply  # apply inherited from Function
    x_mod = modulus(x)

    Parameters
    ---------
    x : tensor
        The complex tensor (i.e., whose last dimension is two) whose modulus
        we want to compute.

    Returns
    -------
    output : tensor
        A tensor of same size as the input tensor, except for the last
        dimension, which is removed. This tensor is differentiable with respect
        to the input in a stable fashion (so gradent of the modulus at zero is
        zero).
    """

    def __init__(self, backend='skcuda'):
        self.CUDA_NUM_THREADS = 1024
        self.backend = backend

    def get_blocks(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, input):

        if not input.is_cuda and self.backend=='skcuda':
            raise RuntimeError('Use the torch backend for CPU tensors!')

        out = input.new(input.size())
        input = input.contiguous()

        if not is_complex(input):
            raise TypeError('The input and outputs should be complex.')

        kernel = """
        extern "C"
        __global__ void abs_complex_value(const ${dtype} * x, ${dtype}2 * z, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        z[i] = make_${dtype}2(normf(2, x + 2*i), 0);

        }
        """
        fabs = load_kernel('abs_complex_value', kernel, dtype=get_dtype(input))
        fabs(grid=(self.get_blocks(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[input.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out

modulus = Modulus()

def modulus_complex(x):
    """Compute the complex modulus

    Computes the modulus of x and stores the result in a complex tensor of the
    same size, with the real part equal to the modulus and the imaginary part
    equal to zero.

    Parameters
    ----------
    x : tensor
        A complex tensor (that is, whose last dimension is equal to 2).

    Returns
    -------
    res : tensor
        A tensor with the same dimensions as x, such that res[..., 0] contains
        the complex modulus of x, while res[..., 1] = 0.
    """
    return modulus(x)

class SubsampleFourier(object):
    """Subsampling in the Fourier domain

    Subsampling in the temporal domain amounts to periodization in the Fourier
    domain, so the input is periodized according to the subsampling factor.

    Usage
    -----
    sub_fourier = SubsampleFourier()
    res = sub_fourier(x, 8)

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
    def __init__(self, backend='skcuda'):
        self.block = (1024, 1, 1)
        self.backend = backend

    def get_blocks(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, input, k):
        if not input.is_cuda and self.backend == 'skcuda':
            raise RuntimeError('Use the torch backend for cpu tensors!')

        out = input.new(input.size(0), input.size(1), input.size(2) // k, 2)

        if not is_complex(input):
            raise (TypeError('The input and outputs should be complex'))

        input = input.contiguous()

        kernel = '''
        #define NT ${T} / ${k}
        extern "C"
        __global__ void periodize(const ${dtype}2 *input, ${dtype}2 *output)
        {
          int tx = blockIdx.x * blockDim.x + threadIdx.x;
          int ty = blockIdx.y * blockDim.y + threadIdx.y;

          if(tx >= NT || ty >= ${B})
            return;
          input += ty * ${T} + tx;
          ${dtype}2 res = make_${dtype}2(0.f, 0.f);

            for (int i=0; i<${k}; ++i)
            {
              const ${dtype}2 &c = input[i * NT];
              res.x += c.x;
              res.y += c.y;
            }
          res.x /= ${k};
          res.y /= ${k};
          output[ty * NT + tx] = res;
        }
        '''
        B = input.size(0) * input.size(1)
        T = input.size(2)
        periodize = load_kernel('periodize', kernel, B=B, T=T, k=k, dtype=get_dtype(input))
        grid = (self.get_blocks(out.size(-2), self.block[0]),
                self.get_blocks(out.nelement() // (2*out.size(-2)), self.block[1]),
                1)
        periodize(grid=grid, block=self.block, args=[input.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out

subsamplefourier = SubsampleFourier()

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
    return subsamplefourier(x,k)

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
    res = F.pad(x.unsqueeze(2),
                (pad_left, pad_right, 0, 0),
                mode=mode, value=value).squeeze(2)
    return res

def pad(x, pad_left=0, pad_right=0, to_complex=True):
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
    to_complex : boolean, optional
        Whether to map the resulting padded tensor to a complex type (seen
        as a real number). Defaults to True.

    Returns
    -------
    output : tensor
        A padded signal, possibly transformed into a four-dimensional tensor
        with the last axis of size 2 if to_complex is True (this axis
        corresponds to the real and imaginary parts).
    """
    output = pad_1d(x, pad_left, pad_right, mode='reflect')
    if to_complex:
        output = torch.stack((output, torch.zeros_like(output)), dim=-1)
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

def real(x):
    """Real part of complex tensor

    Takes the real part of a complex tensor, where the last axis corresponds
    to the real and imaginary parts.

    Parameters
    ----------
    x : tensor
        A complex tensor (that is, whose last dimension is equal to 2).

    Returns
    -------
    x_real : tensor
        The tensor x[..., 0] which is interpreted as the real part of x.
    """
    return x[..., 0]

def fft1d_c2c(x):
    """Compute the 1D FFT of a complex signal

    Input
    -----
    x : tensor
        A tensor of size (..., T, 2), where x[..., 0] is the real part and
        x[..., 1] is the imaginary part.

    Returns
    -------
    x_f : tensor
        A tensor of the same size as x containing its Fourier transform in the
        standard PyTorch FFT ordering.
    """
    return torch.fft(x, signal_ndim=1)

def ifft1d_c2c(x):
    """Compute the normalized 1D inverse FFT of a complex signal

    Input
    -----
    x_f : tensor
        A tensor of size (..., T, 2), where x_f[..., 0] is the real part and
        x[..., 1] is the imaginary part. The frequencies are assumed to be in
        the standard PyTorch FFT ordering.

    Returns
    -------
    x : tensor
        A tensor of the same size of x_f containing the normalized inverse
        Fourier transform of x_f.
    """
    return torch.ifft(x, signal_ndim=1)
