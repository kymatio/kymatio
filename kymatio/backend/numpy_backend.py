import numpy
import scipy.fft


class NumpyBackend:
    _np = numpy
    _fft = scipy.fft

    name = 'numpy'

    @staticmethod
    def input_checks(x):
        if x is None:
            raise TypeError('The input should be not empty.')

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
    def concatenate(cls, arrays, axis=-2):
        return cls._np.stack(arrays, axis=axis)

    @classmethod
    def concatenate_v2(cls, arrays, axis=1, stack=False):
        if stack:
            # emulate `np.stack`
            if axis < 0:
                slc = (slice(None),) * (arrays[0].ndim + axis + 1) + (None,)
            else:
                slc = (slice(None),) * axis + (None,)
            arrays = [a[slc] for a in arrays]
        return cls._np.concatenate(arrays, axis=axis)

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

        sa, sb = A.shape, B.shape
        # last dims equal, or last except *the* last if the last is 1 in A or B
        if not ((sa[-B.ndim:] == sb) or
                ((sa[-1] == 1 or sb[-1] == 1) and (sa[-B.ndim:-1] == sb[:-1]))):
            raise RuntimeError('The inputs are not compatible for '
                               'multiplication (%s and %s).' % (sa, sb))

        if not cls._is_complex(B) and not cls._is_real(B):
            raise TypeError('The second input must be complex or real.')

        return A * B

    @classmethod
    def sqrt(cls, x, dtype=None):
        return cls._np.sqrt(x, dtype=dtype)

    @classmethod
    def mean(cls, x, axis=-1, keepdims=True):
        return x.mean(axis, keepdims=keepdims)

    @classmethod
    def conj(cls, x, inplace=False):
        if inplace and cls._is_complex(x):
            out = cls._np.conj(x, out=x)
        elif not inplace:
            out = (cls._np.conj(x) if cls._is_complex(x) else
                   x)
        else:
            out = x
        return out

    @classmethod
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return cls._np.zeros(shape, dtype=ref.dtype)

    @classmethod
    def reshape(cls, x, shape):
        return x.reshape(*shape)

    @classmethod
    def transpose(cls, x, axes):
        return x.transpose(*axes)

    @classmethod
    def assign_slice(cls, x, x_slc, slc):
        x[slc] = x_slc
        return x
