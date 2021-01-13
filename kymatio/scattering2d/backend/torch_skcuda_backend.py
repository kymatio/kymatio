# Authors: Edouard Oyallon, Sergey Zagoruyko, Muawiz Chaudhary

from collections import namedtuple
import torch
import cupy
from string import Template


BACKEND_NAME = 'torch_skcuda'

from ...backend.torch_backend import _is_complex
from ...backend.torch_skcuda_backend import cdgmm

# As of v8, cupy.util has been renamed cupy._util.
if hasattr(cupy, '_util'):
    memoize = cupy._util.memoize
else:
    memoize = cupy.util.memoize

@memoize(for_each_device=True)
def _load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

Stream = namedtuple('Stream', ['ptr'])

def _get_dtype(t):
    dtypes = {torch.float32: 'float',
              torch.float64: 'double'}

    return dtypes[t.dtype]

class SubsampleFourier(object):
    """Subsampling of a 2D image performed in the Fourier domain.

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor
             Torch tensor with at least 5 dimensions, the last being the real
             and imaginary parts. Ideally, the last dimension should be a
             power of 2 to avoid errors.
        k : int
            Integer such that x is subsampled by k along the spatial variables.

        Raises
        ------
        RuntimeError
            In the event that x is not contiguous.
        TypeError
            In the event that x is on CPU or the input is not complex.

        Returns
        -------
        out : tensor
            Tensor such that its fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k)].

    """
    def __init__(self):
        self.block = (32, 32, 1)

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, x, k):
        if not x.is_cuda:
            raise TypeError('Use the torch backend (without skcuda) for CPU tensors.')

        batch_shape = x.shape[:-3]
        signal_shape = x.shape[-3:]
        x = x.view((-1,) + signal_shape)

        out = x.new(size=[x.shape[0], x.shape[1] // k, x.shape[2] // k, 2])

        if not _is_complex(x):
            raise TypeError('The x should be complex.')

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
        B = x.shape[0]
        W = x.shape[2]
        H = x.shape[1]

        periodize = _load_kernel('periodize', kernel, B=B, H=H, W=W, k=k, Dtype=_get_dtype(x))
        grid = (self.GET_BLOCKS(out.shape[1], self.block[0]),
                self.GET_BLOCKS(out.shape[2], self.block[1]),
                self.GET_BLOCKS(out.shape[0], self.block[2]))
        periodize(grid=grid, block=self.block, args=[x.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        out = out.reshape(batch_shape + out.shape[-3:])
        return out

class Modulus(object):
    """This class implements a modulus transform for complex numbers.

        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)

        Parameters
        ---------
        x : tensor
            Complex torch tensor.

        Raises
        ------
        RuntimeError
            In the event that x is not contiguous.
        TypeError
            In the event that x is on CPU or the input is not complex.

        Returns
        -------
        output : tensor
            A tensor with the same dimensions as x, such that output[..., 0]
            contains the complex modulus of x, while output[..., 1] = 0.

    """
    def __init__(self):
        self.CUDA_NUM_THREADS = 1024

    def GET_BLOCKS(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, x):
        if not x.is_cuda:
            raise TypeError('Use the torch backend (without skcuda) for CPU tensors.')

        out = x.new(x.shape)

        if not _is_complex(x):
            raise TypeError('The inputs should be complex.')

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
        fabs = _load_kernel('abs_complex_value', kernel, Dtype=_get_dtype(x))
        fabs(grid=(self.GET_BLOCKS(int(out.nelement()) // 2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[x.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


from .torch_backend import unpad
from .torch_backend import Pad
from .torch_backend import fft
from .torch_backend import concatenate

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])
backend.name = 'torch_skcuda'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.concatenate = lambda x: concatenate(x, -3)
