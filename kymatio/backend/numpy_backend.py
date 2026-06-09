import numpy
import scipy.fft

from .base_backend import BaseBackend, backend_types, backend_basic_math, backend_array

numpy_backend_function_names = [
        "nan",
        "moveaxis",
        "trace",
        "copy",
        "transpose",
        "arange",
        "flip",
        "kron",
        #"concatenate",
        "max",
        "mean",
        "sum",
        "argmin",
        "argmax",
        "sign",
        "stack",
        "conj",
        "diag",
        "log",
        "log2",
        "tensordot",
        "argsort",
        "sort",
        "dot",
        "shape",
    ]

class NumpyBackendType(type):
    def __getattr__(cls, name):
        if name in backend_types:
            return getattr(cls._np, name)
        elif name in backend_basic_math:
            return getattr(cls._np, name)
        elif name in backend_array:
            return getattr(cls._np, name)
        elif name in numpy_backend_function_names:
            return getattr(cls._np, name)
        


class NumpyBackend(BaseBackend, metaclass=NumpyBackendType):
    _np = numpy
    _fft = scipy.fft

    name = 'numpy'

    @staticmethod
    def input_checks(x):
        if x is None:
            raise TypeError('The input should be not empty.')

        #want to make sure that we have either a numpy or numpy-like arrays.
        #since each numpy-like implements its own distinct ndarray, we need to
        #check if numpy
        if (not isinstance(x, numpy.ndarray)) and (not isinstance(x, _np.ndarray)):
            raise TypeError(f'The input should be a numpy array, got type {type(x)}')

    @classmethod
    def complex_check(cls, x):
        if not cls._is_complex(x):
            raise TypeError('The input should be complex.')

    @classmethod
    def real_check(cls, x):
        if not cls._is_real(x):
            raise TypeError('The input should be real.')

    @classmethod
    def _is_complex(cls, x):
        return (x.dtype == cls._np.complex64) or (x.dtype == cls._np.complex128)

    @classmethod
    def _is_real(cls, x):
        return (x.dtype == cls._np.float32) or (x.dtype == cls._np.float64)

    @classmethod
    def stack(cls, arrays, dim=1):
        return cls._np.stack(arrays, axis=dim)

    @classmethod
    def modulus(cls, x):
        """
            This function implements a modulus transform for complex numbers.

            Usage
            -----
            x_mod = modulus(x)

            Parameters
            ---------
            x: input complex tensor.

            Returns
            -------
            output: a real tensor equal to the modulus of x.

        """
        return cls._np.abs(x)

    @classmethod
    def cdgmm(cls, A, B):
        """
            Complex pointwise multiplication between (batched) tensor A and tensor B.

            Parameters
            ----------
            A : tensor
                A is a complex tensor of size (B, C, M, N, 2)
            B : tensor
                B is a complex tensor of size (M, N) or real tensor of (M, N)
            inplace : boolean, optional
                if set to True, all the operations are performed inplace

            Returns
            -------
            C : tensor
                output tensor of size (B, C, M, N, 2) such that:
                C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]

        """
        if not cls._is_complex(A):
            raise TypeError('The first input must be complex.')

        if A.shape[-len(B.shape):] != B.shape[:]:
            raise RuntimeError('The inputs are not compatible for '
                               'multiplication.')

        if not cls._is_complex(B) and not cls._is_real(B):
            raise TypeError('The second input must be complex or real.')

        return A * B

    @staticmethod
    def reshape_input(x, signal_shape):
        return x.reshape((-1, 1) + signal_shape)

    @staticmethod
    def reshape_output(S, batch_shape, n_kept_dims):
        new_shape = batch_shape + S.shape[-n_kept_dims:]
        return S.reshape(new_shape)

    # A few more, for backend enhancement, copied from tensorly
    @staticmethod
    def to_numpy(tensor):
        return np.copy(tensor)

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)
