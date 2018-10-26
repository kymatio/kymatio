from collections import defaultdict, namedtuple
import torch
from skcuda import cublas
import cupy
from string import Template
from torch.legacy.nn import SpatialReflectionPadding as pad_function

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

# This function copies and view the real to complex
def pad(input, pre_pad):
    if(pre_pad):
        output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)
        output.narrow(output.ndimension()-1, 0, 1).copy_(input)
    else:
        out_ = self.padding_module.updateOutput(input)
        output = input.new(*(out_.size() + (2,))).fill_(0)
        output.select(4, 0).copy_(out_)
    return output

def unpad(self, in_):
    return in_[..., 1:-1, 1:-1]

class Periodize(object):
    """This class builds a wrapper to the periodiziation kernels and cache them.
        """
    def __init__(self, backend='skcuda'):
        self.block = (32, 32, 1)
        self.backend = backend

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, input, k):
        if not input.is_cuda and self.backend == 'skcuda':
            raise RuntimeError('Use the torch backend for cpu tensors!')

        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)

        if not iscomplex(input):
            raise (TypeError('The input and outputs should be complex'))

        input = input.contiguous()

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
        B = input.nelement() // (2*input.size(-2) * input.size(-3))
        W = input.size(-2)
        H = input.size(-3)
        k = input.size(-2) // out.size(-2)
        periodize = load_kernel('periodize', kernel, B=B, H=H, W=W, k=k, Dtype=getDtype(input))
        grid = (self.GET_BLOCKS(out.size(-3), self.block[0]),
                self.GET_BLOCKS(out.size(-2), self.block[1]),
                self.GET_BLOCKS(out.nelement() // (2*out.size(-2) * out.size(-3)), self.block[2]))
        periodize(grid=grid, block=self.block, args=[input.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


class Modulus(object):
    """This class builds a wrapper to the moduli kernels and cache them.
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




class Fft(object):
    """This class builds a wrapper to the FFTs kernels and cache them.

    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
        """
    def __call__(self, input, direction='C2C', inverse=False):
        if direction == 'C2R':
            inverse = True

        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        if direction == 'C2R':
            output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)
        elif direction == 'C2C':
            if inverse:
                output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
            else:
                output = torch.fft(input, 2, normalized=False)

        return output




def cdgmm(A, B, inplace=False):
    """This function uses the C-wrapper to use cuBLAS.
        """
    A, B = A.contiguous(), B.contiguous()
    if A.size()[-3:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 3:
        raise RuntimeError('The filters must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    if not A.is_cuda:
        raise RuntimeError('Use the torch backend for cpu tensors!')

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




