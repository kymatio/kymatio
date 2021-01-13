# Authors: Edouard Oyallon, Joakim Anden

import torch
import cupy
from collections import namedtuple
from string import Template

BACKEND_NAME = 'torch_skcuda'

# As of v8, cupy.util has been renamed cupy._util.
if hasattr(cupy, '_util'):
    memoize = cupy._util.memoize
else:
    memoize = cupy.util.memoize

@memoize(for_each_device=True)
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

def _is_complex(input):
    return input.shape[-1] == 2

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

    def __call__(self, x):
        if not x.is_cuda and self.backend=='skcuda':
            raise TypeError('Use the torch backend (without skcuda) for CPU tensors.')

        out = x.new(x.shape)

        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')

        if not _is_complex(x):
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
        fabs = load_kernel('abs_complex_value', kernel, dtype=get_dtype(x))
        fabs(grid=(self.get_blocks(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[x.data_ptr(), out.data_ptr(), out.numel() // 2],
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
    norm : tensor
        A tensor with the same dimensions as x, such that norm[..., 0] contains
        the complex modulus of x, while norm[..., 1] = 0.
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

    def __call__(self, x, k):
        if not x.is_cuda and self.backend == 'skcuda':
            raise TypeError('Use the torch backend (without skcuda) for CPU tensors.')

        if not _is_complex(x):
            raise TypeError('The input and outputs should be complex.')

        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')

        out = x.new(x.shape[0], x.shape[1], x.shape[2] // k, 2)

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
        B = x.shape[0] * x.shape[1]
        T = x.shape[2]
        periodize = load_kernel('periodize', kernel, B=B, T=T, k=k, dtype=get_dtype(x))
        grid = (self.get_blocks(out.shape[-2], self.block[0]),
                self.get_blocks(out.nelement() // (2*out.shape[-2]), self.block[1]),
                1)
        periodize(grid=grid, block=self.block, args=[x.data_ptr(), out.data_ptr()],
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


from .torch_backend import real, cdgmm, unpad, pad, pad_1d, concatenate, fft

backend = namedtuple('backend', ['name', 'modulus_complex', 'subsample_fourier', 'real', 'unpad', 'fft', 'concatenate'])

backend.name = 'torch_skcuda'
backend.modulus_complex = modulus_complex
backend.subsample_fourier = subsample_fourier
backend.real = real
backend.cdgmm = cdgmm
backend.unpad = unpad
backend.pad = pad
backend.pad_1d = pad_1d
backend.fft = fft
backend.concatenate = lambda x: concatenate(x, -2)
