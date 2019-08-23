# Authors: Edouard Oyallon, Sergey Zagoruyko

from collections import namedtuple
import torch
from skcuda import cublas
import cupy
from string import Template
from torch.nn import ReflectionPad2d



BACKEND_NAME = 'torch_skcuda'

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

def iscomplex(x):
    return x.size(-1) == 2

def isreal(x):
    return x.size(-1) == 1

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
    def __init__(self):
        self.block = (32, 32, 1)

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, x, k):
        if not x.is_cuda:
            raise TypeError('Use the torch backend (without skcuda) for cpu tensors!')

        batch_shape = x.shape[:-3]
        signal_shape = x.shape[-3:]
        x = x.view((-1,) + signal_shape)

        out = x.new(size=[x.size(0), x.size(1) // k, x.size(2) // k, 2])

        if not iscomplex(x):
            raise TypeError('The x and outputs should be complex.')

        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')

        kernel = '''
        #define NW ${W} / ${k}
        #define NH ${H} / ${k}
        extern "C"
        __global__ void periodize(const ${Dtype}2 *input, ${Dtype}2 *output)
        {
          int tx = blockIdx.x * blockDim.x + threadIdx.x;
          int ty = blockIdx.y * blockDim.y + threadIdx.y;
          int tz = blockIdx.z * blockDim.z + threadIdx.z;
          if(tx >= NW || ty >= NH || tz >= ${B})
            return;
          input += tz * ${H} * ${W} + ty * ${W} + tx;
          ${Dtype}2 res = make_${Dtype}2(0.f, 0.f);
          for (int j=0; j<${k}; ++j)
            for (int i=0; i<${k}; ++i)
            {
              const ${Dtype}2 &c = input[j * NH * ${W} + i * NW];
              res.x += c.x;
              res.y += c.y;
            }
          res.x /= ${k} * ${k};
          res.y /= ${k} * ${k};
          output[tz * NH * NW + ty * NW + tx] = res;
        }
        '''
        B = x.size(0)
        W = x.size(2)
        H = x.size(1)

        periodize = load_kernel('periodize', kernel, B=B, H=H, W=W, k=k, Dtype=getDtype(x))
        grid = (self.GET_BLOCKS(out.size(1), self.block[0]),
                self.GET_BLOCKS(out.size(2), self.block[1]),
                self.GET_BLOCKS(out.size(0), self.block[2]))
        periodize(grid=grid, block=self.block, args=[x.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        out = out.reshape(batch_shape + out.shape[-3:])
        return out


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
    def __init__(self):
        self.CUDA_NUM_THREADS = 1024

    def GET_BLOCKS(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, x):
        if not x.is_cuda:
            raise TypeError('Use the torch backend (without skcuda) for cpu tensors!')

        out = x.new(x.size())

        if not iscomplex(x):
            raise TypeError('The input and outputs should be complex')

        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')

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
        fabs = load_kernel('abs_complex_value', kernel, Dtype=getDtype(x))
        fabs(grid=(self.GET_BLOCKS(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[x.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


def cdgmm(A, B, inplace=False):
    """
        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2)
        B : tensor
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1)
        inplace : boolean, optional
            if set to True, all the operations are performed inplace

        Returns
        -------
        C : tensor
            output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]
    """
    if not iscomplex(A):
        raise TypeError('The input must be complex, indicated by a last '
                        'dimension of size 2')

    if B.ndimension() != 3:
        raise RuntimeError('The filter must be a 3-tensor, with a last '
                           'dimension of size 1 or 2 to indicate it is real '
                           'or complex, respectively')

    if not iscomplex(B) and not isreal(B):
        raise TypeError('The filter must be complex or real, indicated by a '
                        'last dimension of size 2 or 1, respectively')

    if A.size()[-3:-1] != B.size()[-3:-1]:
        raise RuntimeError('The filters are not compatible for multiplication!')

    if A.dtype is not B.dtype:
        raise TypeError('A and B must be of the same dtype.')

    if not A.is_cuda or not B.is_cuda:
        raise TypeError('A and B must be cuda tensors.')

    if A.device.index != B.device.index:
        raise TypeError('A and B must be on the same GPU!')

    if isreal(B):
        if inplace:
            return A.mul_(B)
        else:
            return A * B
    else:
        if not A.is_contiguous() or not B.is_contiguous():
            raise RuntimeError('A and B should be contiguous.')

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

from .torch_backend import unpad
from .torch_backend import Pad
from .torch_backend import fft
from .torch_backend import finalize

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'finalize'])
backend.name = 'torch_skcuda'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.finalize = finalize
