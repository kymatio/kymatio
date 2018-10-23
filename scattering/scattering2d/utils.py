"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""
from collections import defaultdict, namedtuple
import torch
from skcuda import cublas
import cupy
from string import Template


@cupy.util.memoize(for_each_device=True)
#load the kernel with the string representing the code
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)          #substitute the additional arguments into the string representing the code
    kernel_code = cupy.cuda.compile_with_cache(code)    #compile the code for cuda
    return kernel_code.get_function(kernel_name)        #return a function with the load_kernel arguments inserted


Stream = namedtuple('Stream', ['ptr'])

#get type of cuda tensor
def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

#check if input is complex by testing of the last dimension of the array has 2 channels for real and imaginary respectively
def iscomplex(input):
    return input.size(-1) == 2


class Periodize(object):
    """This class periodizes an object, i.e. it categorizes the data into discrete, quantified blocks with respect to their frequency.
    #QUESTION is this the 'rounding' operation that bins multiple frequencies in the frequency domain together?
    It builds a wrapper to the periodiziation kernels and cache them.
    """
    def __init__(self, backend='skcuda'):
        self.block = (32, 32, 1)
        self.backend = backend

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    #apply the periodization
    def __call__(self, input, k):
        #QUESTION: is k the number of bins for the periodization?

        #check that input and backend type align
        if not input.is_cuda and self.backend == 'skcuda':
            raise RuntimeError('Use the torch backend for cpu tensors!')

        #define an output tensor.
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)

        #define output tensor if backend is torch
        if self.backend == 'torch':
            y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)

            out = y.mean(4, keepdim=False).mean(2, keepdim=False)       #reduce dimension 4 and 2 by taking their respective mean
            return out

        #check that input is complex
        if not iscomplex(input):
            raise (TypeError('The input and outputs should be complex'))

        #make input contiguous in memory
        input = input.contiguous()

        #define kernel operation that is used on GPU
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

        #define additional arguments for the load kernel function
        B = input.nelement() // (2*input.size(-2) * input.size(-3))         #define breadth
        W = input.size(-2)                                                  #define width
        H = input.size(-3)                                                  #define height
        k = input.size(-2) // out.size(-2)                                  #define number of bins for periodization
        #use the previously defined load_kernel function
        periodize = load_kernel('periodize', kernel, B=B, H=H, W=W, k=k, Dtype=getDtype(input))
        grid = (self.GET_BLOCKS(out.size(-3), self.block[0]),
                self.GET_BLOCKS(out.size(-2), self.block[1]),
                self.GET_BLOCKS(out.nelement() // (2*out.size(-2) * out.size(-3)), self.block[2]))
        #apply the just define periodize function and load the result into the output tensor
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

        #check that input and backend align
        if not input.is_cuda and self.backend=='skcuda':
            raise RuntimeError('Use the torch backend for cpu tensors!')

        #if the backend is torch use the norm
        if self.backend=='torch':
            norm = input.norm(p=2, dim=-1, keepdim=True)                #take the 2norm of the input
            return torch.cat([norm, torch.zeros_like(norm)], -1)        #concatenate the 2norm with an array of zeros to kill the phase


        out = input.new(input.size())           #define a new output tensor with same shape as input
        input = input.contiguous()              #make the input contiguous in memory

        #check that input is complex
        if not iscomplex(input):
            raise TypeError('The input and outputs should be complex')

        #define the kernel operation to calculate the modulus
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

        #use load_kernel as defined previously
        fabs = load_kernel('abs_complex_value', kernel, Dtype=getDtype(input))
        #apply the function fabs and store result in out tensor
        fabs(grid=(self.GET_BLOCKS(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[input.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out




class Fft(object):
    """This class builds a wrapper to the FFTs kernels and cache them.
    FFT stands for fast fourier transform.
    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
    for cufft documentation see: https://docs.nvidia.com/cuda/cufft/index.html
    """
    def __call__(self, input, direction='C2C', inverse=False):
        """
        input is a tensor representing a complex matrix on which the fft should be applied
        direction is either C2R for complex to real or C2C for complex to complex
        inverse is a boolean determining whether the inverse fourier transform should be applied
        """
        #if the direction is C2R we want to set inverse True. This makes sure that we are always starting our calculation with a complex matrix
        if direction == 'C2R':
            inverse = True

        #check that the input is a complex matrix
        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        #check that input is contiguous in memory
        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        #compute the inverse fft of the input
        if direction == 'C2R':          # QUESTION: why do you again ask for the direction? Can't you just allocate the cache in the first if statement?
            output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)

        #compute the fft between two complex matrices. Depending on whether inverse is True or False, either forward or inverse transform
        elif direction == 'C2C':
            if inverse:
                output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
            else:
                output = torch.fft(input, 2, normalized=False)

        return output




def cdgmm(A, B, backend='skcuda', inplace=False):
    """This function is a complex matrix multiplicationuses.
    it the C-wrapper to use cuBLAS.
    A, B are matrices with complex entries.
    backend denotes the backend for computation
    inplace denotes whether the computation should be done inplace in memory
    """

    #make the matrices A and B contiguous in memory
    A, B = A.contiguous(), B.contiguous()

    #check that the filers are compatible for multiplication
    if A.size()[-3:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    #check whether A and B are complex matrices
    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    #check that the dimension of B is not 3 and therefore make sure the filters are complex
    if B.ndimension() != 3:
        raise RuntimeError('The filters must be simply a complex array!')

    #check that A and B have the same type, i.e. a FloatTensor in pytorch
    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    #check that backend and type of tensors align
    if not A.is_cuda and backend=='skcuda':
        raise RuntimeError('Use the torch backend for cpu tensors!')

    #use the pytorch syntax of creating Tensors
    if backend=='torch':
        C = A.new(A.size())

                                                                            #.view() appearantly is equivalent to .reshape() in numpy
        A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))        #A_r are the real values of A but flattened
        A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))        #A_i are the imaginary values of A but flattened

                                                                                                        #.unsqueeze() adds a dimension to the beginning of the tensor
                                                                                                        #.expand_as() expands the tensor to have the same size as the argument tensor
        B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)             #B_r are the real values of B with the same shape as A_r
        B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)            #B_i are the imaginary parts of B with the same shape as A_i

                                                                                       #[...,0] is a pythonic way of specifiying that a certain operation calculated in place
        C[..., 0].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_r - A_i * B_i           #first column of C denotes the real part of the multiplication of A and B
        C[..., 1].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_i + A_i * B_r           #second column of C denotes the imaginary part of the multiplication of A and B

        return C if not inplace else A.copy_(C)

    #else in this case means using a cuda backend
    else:
        C = A.new(A.size()) if not inplace else A                       #create a new Tensor representing complex numbers where results of the multiplication are stored
        m, n = B.nelement() // 2, A.nelement() // B.nelement()          #m is the half the number of elements in B, n is the number of elements in A devided by the number of elements in B
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()                       #returns the pointer to the current cublas handler
        stream = torch.cuda.current_stream()._as_parameter_             #returns a currently selected stream
        cublas.cublasSetStream(handle, stream)                          #sets the cublas stream. If not set, stream is NULL
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        #for documentation see: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm
        #handle = selected cublas handler, 'l' selects left multiplication, m and n are number of rows and columns of matrix A and C respectively,
        #A.data_ptr() and B.data_ptr() return the address of the first element of self tensor, incx sets the stride,
        #lda and ldc represent the leading dimension of the two dimensional arrays A and B
        return C
