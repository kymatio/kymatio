"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""
from collections import defaultdict, namedtuple

import torch
from skcuda import cublas, cufft
import numpy as np
import cupy
from string import Template


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


class Periodize(object):
    """This class builds a wrapper to the periodiziation kernels and cache them.
        """
    def __init__(self, jit=True):
        self.block = (32, 32, 1)
        self.jit = jit

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, input, k):
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)

        if not self.jit or not input.is_cuda:
            y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)

            out = y.mean(4, keepdim=False).mean(2, keepdim=False)
            return out

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
    def __init__(self, jit=True):
        self.CUDA_NUM_THREADS = 1024
        self.jit = jit

    def GET_BLOCKS(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, input):
        if not self.jit or not input.is_cuda:
            norm = input.norm(p=2, dim=-1, keepdim=True)
            return torch.cat([norm, norm.new(norm.size()).zero_()], -1)

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




"""
class Fft(object):
    """"""This class builds a wrapper to the FFTs kernels and cache them.

    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
        """"""
    def __call__(self, input, direction='C2C', inverse=False, inplace=False):
        if direction == 'C2R':
            inverse = True

        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        output = []
        input_ = input#.view(input.size(0)*input.size(1),input.size(2),input.size(3),input.size(4))
        if direction == 'C2R':
            output = torch.irfft(input_,2,normalized=False, onesided=False)
        elif direction == 'C2C':
            if inverse:
                output = torch.ifft(input_, 2, normalized=False)
            else:
                output = torch.fft(input_, 2, normalized=False)

        output = output * output.size(3)*output.size(2)
        #print(output.size())
        #print(input.size())
        #output = output.view(input.size(0), input.size(1), input.size(2), input.size(3), input.size(4))
        return output
"""


class Fft(object):
    """This class builds a wrapper to the FFTs kernels and cache them.
    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
        """

    def __init__(self):
        self.fft_cache = defaultdict(lambda: None)

    def buildCache(self, input, type):
        k = input.ndimension() - 3
        n = np.asarray([input.size(k), input.size(k+1)], np.int32)
        batch = input.nelement() // (2*input.size(k) * input.size(k + 1))
        idist = input.size(k) * input.size(k + 1)
        istride = 1
        ostride = istride
        odist = idist
        rank = 2
        plan = cufft.cufftPlanMany(rank, n.ctypes.data, n.ctypes.data, istride,
                                   idist, n.ctypes.data, ostride, odist, type, batch)
        self.fft_cache[(input.size(), type, input.get_device())] = plan

    def __del__(self):
        for keys in self.fft_cache:
            try:
                cufft.cufftDestroy(self.fft_cache[keys])
            except:
                pass

    def __call__(self, input, direction='C2C', inplace=False, inverse=False):
        if direction == 'C2R':
            inverse = True

        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                input_np = input[..., 0].numpy() + 1.0j * input[..., 1].numpy()
                f = lambda x: np.stack((np.real(x), np.imag(x)), axis=len(x.shape))
                out_type = input.numpy().dtype

                if direction == 'C2R':
                    out = np.real(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    return torch.from_numpy(out)

                if inplace:
                    if inverse:
                        out = f(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    else:
                        out = f(np.fft.fft2(input_np)).astype(out_type)
                    input.copy_(torch.from_numpy(out))
                    return
                else:
                    if inverse:
                        out = f(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    else:
                        out = f(np.fft.fft2(input_np)).astype(out_type)
                    return torch.from_numpy(out)

        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        if direction == 'C2R':
            output = input.new(input.size()[:-1])
            if(self.fft_cache[(input.size(), cufft.CUFFT_C2R, input.get_device())] is None):
                self.buildCache(input, cufft.CUFFT_C2R)
            cufft.cufftExecC2R(self.fft_cache[(input.size(), cufft.CUFFT_C2R, input.get_device())],
                               input.data_ptr(), output.data_ptr())

            z = torch.irfft(input, 2, normalized=False, onesided=False) - output
            print('C2R')
            print(z.abs().max())
            return output
        elif direction == 'C2C':
            output = input.new(input.size()) if not inplace else input
            flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
            if (self.fft_cache[(input.size(), cufft.CUFFT_C2C, input.get_device())] is None):
                self.buildCache(input, cufft.CUFFT_C2C)
            cufft.cufftExecC2C(self.fft_cache[(input.size(), cufft.CUFFT_C2C, input.get_device())],
                               input.data_ptr(), output.data_ptr(), flag)
            z = torch.irfft(input, 2, normalized=False, onesided=False)
            print(z.size())
            print(output.size())
            z = z - output
            print('C2C')

            print(z.abs().max())
            return output


def cdgmm(A, B, jit=True, inplace=False):
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

    if not jit or not A.is_cuda:
        C = A.new(A.size())

        A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
        A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

        B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
        B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

        C[..., 0].copy_(A_r * B_r - A_i * B_i)
        C[..., 1].copy_(A_r * B_i + A_i * B_r)

        # faster if B is actually real
        #B[...,1] = B[...,0]
        #C = A * B.unsqueeze(0).expand_as(A)
        return C if not inplace else A.copy_(C)
    else:
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
