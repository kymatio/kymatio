import torch
from skcuda import cufft
import numpy as np
from collections import defaultdict

"""
Package used to wrap the CUFFT

The main function to use at the end are fft1d_c2c and ifft1d_c2c_normed
which take as input torch Variable objects and return Variable containing
the FFT, which are differentiable with respect to their inputs.
"""


class FFT1DCache(object):

    def __init__(self):
        """ Interface with forward/backward FFT routines for 1D signals.

        An object from this class caches CUFFT plans to avoid recomputing
        them at runtime. Plans are added dynamically depending on the
        shape of input tensors and the type of operation desired (e.g.
        C2C vs C2R).
        This class only accepts 3D tensors as inputs, and the FFT is
        computed along the last axis.
        It will dynamically choose whether to use CUFFT or numpy depending
        on the device on which the input tensor is stored.
        Forward and backward FFT are supported.
        Note that the backward (or inverse) FFT is assumed to be unnormalized:
        it consists in the adjoint of the forward FFT, so that for an input x,
        FFT̂^{-1}(FFT(x)) = n * x, where n is the temporal size of x

        Example
        -------
        fft1d = FFT1DCache()
        x = torch.randn(128, 1, 4096, 2)
        x_fft = fft1d.c2c(x)

        Parameters
        ----------
        None

        Attributes
        ---------
        fft_cache : dictionary
            cache containing the different fft plans, implemented as
            a dictionary, with triplets (input_shape, type, device)
            for keys, where input_shape is a tuple of integers,
            type a CUFFT type (e.g. cufft.CUFFT_C2C) and device
            the address of the device where the operations should be
            performed.
        """
        self._build()

    def _build(self):
        self.fft_cache = defaultdict(lambda: None)

    def createStorePlanCache(self, x, op_type):
        """
        Creates and stores a CUFFT plan in the cache.

        Builds the plan for given type (type of FFT operation required, e.g.
        cufft.CUFFT_C2C), a given tensor shape and a given device and stores it
        in the cache (self.fft_cache).

        Parameters
        ----------
        x : tensor
            input tensor (torch.Tensor)
        type : cufft operation type
            type of operation to perform (cufft.CUFFT_C2C or cufft.CUFFT_C2R)

        Returns
        -------
        None
        """
        k = x.ndimension() - 2  # last dimension
        n = np.asarray([x.shape[k]], np.int32)   # total dimensionality
        batch = x.nelement() // (2 * x.shape[k])  # size of batch
        # i: prefix for Input
        # o: prefix for Output
        # pointer of size rank that indicates the storage dimensions of
        # the input data in memory
        inembed = n.ctypes.data
        # distance between two successive input elements in the least
        # significant (i.e. innermost) dimension
        idist = x.shape[k]
        # indicates the distance between the first element of two
        # consecutive signals in a batch of the input data
        istride = 1
        # cf above, o = 'output'
        ostride = istride
        # cf above
        onembed = n.ctypes.data
        # cf above
        odist = idist
        # dimension
        rank = 1
        # NB: for the second argument, instead of giving n, we provide the
        # pointer on n. It magically works!
        plan = cufft.cufftPlanMany(
            rank, n.ctypes.data, inembed, istride, idist,
            onembed, ostride, odist, op_type, batch)
        self.fft_cache[(x.shape, op_type, x.get_device())] = plan

    def __del__(self):
        """
        Removes the cache when the object is deleted.

        It recursively calls cufft.cufftDestroy on all entries of the cache.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for keys in self.fft_cache:
            try:
                cufft.cufftDestroy(self.fft_cache[keys])
            except:
                pass

    def c2c_cuda(self, x, inverse=False):
        """
        Performs C2C (Complex to Complex) forward/backward FFT on the input.

        The device to use (CPU or GPU) is dynamically inferred from the
        memory location of the input

        Note that in CUFFT the inverse FFT is unnormalized, so that
        FFT^{-1}(FFT(x)) = n x, where n is the temporal length of x

        Parameters
        ----------
        x : tensor-like
            input 4D tensor (torch.Tensor or torch.cuda.Tensor) of
            size B x C x T x 2
            where 2 represents complex numbers (real and imaginary parts)
        inverse : boolean, optional
            whether the FFT to use should be the backward FFT (True)
            or the forward FFT (False). Defaults to False.

        Returns
        -------
        output : tensor-like
            output 4D tensor on the same device as x and of same size,
            where output[i, j] contains the desired FFT of x[i, j]
        """
        if not(x.shape[-1] == 2):
            raise ValueError('The last dim of input array x should be 2!')
        if not x.is_contiguous():
            raise ValueError("Input array x must be contiguous")
        # create a new tensor on the same device with the same size
        output = x.new(x.shape)
        # adapt the flag depending on the inverse or not
        flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
        # adapt the FFT type to Float or Double
        if isinstance(x, torch.cuda.FloatTensor):
            ffttype = cufft.CUFFT_C2C
            exec_to_use = cufft.cufftExecC2C
        elif isinstance(x, torch.cuda.DoubleTensor):
            ffttype = cufft.CUFFT_Z2Z  # double
            exec_to_use = cufft.cufftExecZ2Z
        else:
            raise ValueError("Unsupported type for x:", type(x))
        # build and store the cache if not existing
        path_input = (x.shape, ffttype, x.get_device())
        if self.fft_cache[path_input] is None:
            self.createStorePlanCache(x, ffttype)
        # Execute the FFT transform and store it in output
        exec_to_use(self.fft_cache[path_input],
                    x.data_ptr(), output.data_ptr(), flag)
        return output

    def c2c_cpu(self, x, inverse=False):
        """
        Performs C2C (Complex to Complex) forward/backward FFT on the input.

        The device to use (CPU or GPU) is dynamically inferred from the
        memory location of the input

        Note that in CUFFT the inverse FFT is unnormalized, so that
        FFT^{-1}(FFT(x)) = n x, where n is the temporal length of x

        Parameters
        ----------
        x : tensor-like
            input 4D tensor (torch.Tensor or torch.cuda.Tensor) of
            size B x C x T x 2
            where 2 represents complex numbers (real and imaginary parts)
        inverse : boolean, optional
            whether the FFT to use should be the backward FFT (True)
            or the forward FFT (False). Defaults to False.

        Returns
        -------
        output : tensor-like
            output 4D tensor on the same device as x and of same size,
            where output[i, j] contains the desired FFT of x[i, j]
        """
        # check the last dimension
        if not(x.shape[-1] == 2):
            raise ValueError('The last dim of input array x should be 2!')
        # take it to numpy
        x_np = x.numpy()
        # convert to complex
        if x_np.dtype == 'float32':
            original_type = 'float32'
            x_np.dtype = 'complex64'
        elif x_np.dtype == 'float64':
            original_type = 'float64'
            x_np.dtype = 'complex128'
        else:
            raise ValueError('c2c_cpu supports only floats 32 and 64, but got',
                             x_np.dtype)
        # remove the last axis
        x_np = x_np.reshape(x_np.shape[:-1])
        # perform the FFT operation
        if inverse:  # unnormalized
            res = np.fft.ifft(x_np) * float(x_np.shape[-1])
        else:
            res = np.fft.fft(x_np)
        # Make sure that the types of x_np and res match
        res = np.asarray(res, dtype=x_np.dtype)
        # Separate the real and imaginary parts of res
        res = res[..., np.newaxis]
        res.dtype = original_type
        # move it to torch
        output = torch.from_numpy(res)
        return output

    def c2c(self, x, inverse=False):
        """
        Performs C2C (Complex to Complex) forward/backward FFT on the input.

        The device to use (CPU or GPU) is dynamically inferred from the
        memory location of the input

        Note that in CUFFT the inverse FFT is unnormalized, so that
        FFT^{-1}(FFT(x)) = n x, where n is the temporal length of x

        Parameters
        ----------
        x : tensor-like
            input 4D tensor (torch.Tensor or torch.cuda.Tensor) of
            size B x C x T x 2
            where 2 represents complex numbers (real and imaginary parts)
        inverse : boolean, optional
            whether the FFT to use should be the backward FFT (True)
            or the forward FFT (False). Defaults to False.

        Returns
        -------
        output : tensor-like
            output 4D tensor on the same device as x and of same size,
            where output[i, j] contains the desired FFT of x[i, j]
        """
        if isinstance(x, torch.FloatTensor) or isinstance(x, torch.DoubleTensor):
            return self.c2c_cpu(x, inverse=inverse)
        elif isinstance(x, torch.cuda.FloatTensor) or isinstance(x, torch.cuda.DoubleTensor):
            return self.c2c_cuda(x, inverse=inverse)
        else:
            raise TypeError('Unsupported type for input x' + str(type(x)))


fft1d = FFT1DCache()


class FFT1D_C2C(torch.autograd.Function):
    """
    Differentiable function implementing the cache.
    """

    def forward(self, x):
        return fft1d.c2c(x, inverse=False)

    def backward(self, grad_output):
        return fft1d.c2c(grad_output, inverse=True)


class FFT1D_iC2C(torch.autograd.Function):

    def forward(self, x):
        return fft1d.c2c(x, inverse=True)

    def backward(self, grad_output):
        return fft1d.c2c(grad_output, inverse=False)


def fft1d_c2c(x):
    return FFT1D_C2C()(x)


def ifft1d_c2c(x):
    return FFT1D_iC2C()(x)


def ifft1d_c2c_normed(x):
    factor = 1. / float(x.shape[-2])
    return ifft1d_c2c(x) * factor
