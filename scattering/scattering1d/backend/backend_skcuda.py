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

def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

def iscomplex(input):
    return input.size(-1) == 2

def pad1D(x, pad_left, pad_right, mode='constant', value=0.):
    """
    1D implementation of the padding function for torch tensors

    Parameters
    ----------
    x : tensor_like
        input tensor, 3D with time in the last axis.
    pad_left : int
        amount to add on the left of the tensor (at the beginning
        of the temporal axis)
    pad_right : int
        amount to add on the right of the tensor (at the end
        of the temporal axis)
    mode : string, optional
        Padding mode. Options include 'constant' and
        'reflect'. See the pytorch API for other options.
        Defaults to 'constant'
    value : float, optional
        If mode == 'constant', value to input
        within the padding. Defaults to 0.

    Returns
    -------
    res: the padded tensor
    """
    if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
        if mode == 'reflect':
            raise ValueError('Indefinite padding size (larger than tensor)')
    res = F.pad(x.unsqueeze(2),
                (pad_left, pad_right, 0, 0),
                mode=mode, value=value).squeeze(2)
    return res



class Modulus(object):
    """
        This class implements a modulus transform for complex numbers.

        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)

        Parameters
        ---------
        x: input tensor, with last dimension = 2 for complex numbers

        Returns
        -------
        output: a tensor with imaginary part set to 0, real part set equal to
        the modulus of x.
    """
    def __init__(self, backend='skcuda'):
        self.CUDA_NUM_THREADS = 1024
        self.backend = backend

    def GET_BLOCKS(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, input):

        if not input.is_cuda and self.backend=='skcuda':
            raise RuntimeError('Use the torch backend for cpu tensors!')

        out = input.new(input.size())
        input = input.contiguous()

        if not iscomplex(input):
            raise TypeError('The input and outputs should be complex')

        kernel = """
        extern "C"
        __global__ void abs_complex_value(const ${Dtype} * x, ${Dtype}2 * z, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        z[i] = make_${Dtype}2(normf(2, x + 2*i), 0);

        }
        """
        fabs = load_kernel('abs_complex_value', kernel, Dtype=getDtype(input))
        fabs(grid=(self.GET_BLOCKS(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[input.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out




class SubsampleFourier(object):
    """
        Subsampling of a 2D image performed in the Fourier domain
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least 5 dimensions, the last being the real
             and imaginary parts.
            Ideally, the last dimension should be a power of 2 to avoid errors.
        k : int
            integer such that x is subsampled by 2**k along the spatial variables.

        Returns
        -------
        res : tensor_like
            tensor such that its fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            FFT^{-1}(res)[u1, u2] = FFT^{-1}(x)[u1 * (2**k), u2 * (2**k)]
    """
    def __init__(self, backend='skcuda'):
        self.block = (1024, 1, 1)
        self.backend = backend

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, input, k):
        if not input.is_cuda and self.backend == 'skcuda':
            raise RuntimeError('Use the torch backend for cpu tensors!')

        out = input.new(input.size(0), input.size(1), input.size(2) // k, 2)

        if not iscomplex(input):
            raise (TypeError('The input and outputs should be complex'))

        input = input.contiguous()

        kernel = '''
        #define NT ${T} / ${k}
        extern "C"
        __global__ void periodize(const ${Dtype}2 *input, ${Dtype}2 *output)
        {
          int tx = blockIdx.x * blockDim.x + threadIdx.x;
          int ty = blockIdx.y * blockDim.y + threadIdx.y;

          if(tx >= NT || ty >= ${B})
            return;
          input += ty * ${T} + tx;
          ${Dtype}2 res = make_${Dtype}2(0.f, 0.f);

            for (int i=0; i<${k}; ++i)
            {
              const ${Dtype}2 &c = input[i * NT];
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
        periodize = load_kernel('periodize', kernel, B=B, T=T, k=k, Dtype=getDtype(input))
        grid = (self.GET_BLOCKS(out.size(-2), self.block[0]),
                self.GET_BLOCKS(out.nelement() // (2*out.size(-2)), self.block[1]),
                1)
        periodize(grid=grid, block=self.block, args=[input.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


subsamplefourier = SubsampleFourier()

def subsample_fourier(x, k):
    """
    Subsampling of a vector performed in the Fourier domain

    Subsampling in the temporal domain amounts to periodization
    in the Fourier domain, hence the formula.

    Parameters
    ----------
    x : tensor_like
        input tensor with at least 3 dimensions, the last
        corresponding to time (in the Fourier domain).
        The last dimension should be a power of 2 to avoid errors.
    k : int
        integer such that x is subsampled by 2**k

    Returns
    -------
    res : tensor_like
        tensor such that its fourier transform is the Fourier
        transform of a subsampled version of x, i.e. in
        FFT^{-1}(res)[t] = FFT^{-1}(x)[t * (2**k)]
    """
    return subsamplefourier(x,k)


def pad(x, pad_left=0, pad_right=0, to_complex=True):
    """
    Padding which allows to simultaneously pad in a reflection fashion
    and map to complex if necessary

    Parameters
    ----------
    x : tensor_like
        input tensor, 3D with time in the last axis.
    pad_left : int, optional
        amount to add on the left of the tensor
        (at the beginning of the temporal axis). Defaults to 0
    pad_right : int, optional
        amount to add on the right of the tensor (at the end
        of the temporal axis). Defaults to 0
    to_complex : boolean, optional
        Whether to map the resulting padded tensor
        to a complex type (seen as a real number). Defaults to True

    Returns
    -------
    output : tensor_like
        A padded signal, possibly transformed into a 4D tensor
        with the last axis equal to 2 if to_complex is True (these
        dimensions should be interpreted as real and imaginary parts)
    """
    output = pad1D(x, pad_left, pad_right, mode='reflect')
    if to_complex:
        return torch.cat(
            [output.unsqueeze(-1),
             output.data.new(output.shape + (1,)).fill_(0.)],
            dim=-1)
    else:
        return output


def unpad(x, i0, i1):
    """
    Slices the input tensor at indices between i0 and i1 in the last axis

    Parameters
    ----------
    x : tensor_like
        input tensor (at least 1 axis)
    i0 : int
    i1: int

    Returns
    -------
    x[..., i0:i1]
    """
    return x[..., i0:i1]


def real(x):
    """
    Takes the real part of a 4D tensor x, where the last axis is interpreted
    as the real and imaginary parts.

    Parameters
    ----------
    x : tensor_like

    Returns
    -------
    x[..., 0], which is interpreted as the real part of x
    """
    return x[..., 0]

modulus = Modulus()

def modulus_complex(x):
    """
    Computes the modulus of x as a real tensor and returns a new complex tensor
    with imaginary part equal to 0.

    Parameters
    ----------
    x : tensor_like
        input tensor, should be a 4D tensor with the last axis of size 2

    Returns
    -------
    res : tensor_like
        a tensor with same dimensions as x, such that res[..., 0] contains
        the complex modulus of the input, and res[..., 1] = 0.
    """
    # take the stable modulus
    res = modulus(x)
    return res



def fft1d_c2c(x):
    """
        Computes the fft of a 1d signal.

        Input
        -----
        x : tensor
            the two final sizes must be (T, 2) where T is the time length of the signal
    """
    return torch.fft(x, signal_ndim = 1)


def ifft1d_c2c(x):
    """
        Computes the inverse fft of a 1d signal.

        Input
        -----
        x : tensor
            the two final sizes must be (T, 2) where T is the time length of the signal
    """
    return torch.ifft(x, signal_ndim = 1) * float(x.shape[-2])


def ifft1d_c2c_normed(x):
    """
        Computes the inverse normalized fft of a 1d signal.

        Input
        -----
        x : tensor
            the two final sizes must be (T, 2) where T is the time length of the signal
    """
    return torch.ifft(x, signal_ndim = 1)
