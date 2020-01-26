import numpy as np
import tensorflow as tf

def _iscomplex(x):
    return x.dtype == np.complex64 or x.dtype == np.complex128

def _isreal(x):
    return x.dtype == np.float32 or x.dtype == np.float64

class Modulus(object):
    """This class implements a modulus transform for complex numbers.

    Parameters
    ----------
    x: input complex tensor.

    Returns
    ----------
    output: a real tensor equal to the modulus of x.

    Usage
    ----------
    modulus = Modulus()
    x_mod = modulus(x)
    """
    def __call__(self, x):
        norm = tf.abs(x)
        return tf.cast(norm, tf.complex64)

def real(x):
    """Real part of complex tensor
    Takes the real part of a complex tensor, where the last axis corresponds
    to the real and imaginary parts.
    Parameters
    ----------
    x : tensor
        A complex tensor (that is, whose last dimension is equal to 2).
    Returns
    -------
    x_real : tensor
        The tensor x[..., 0] which is interpreted as the real part of x.
    """
    return tf.math.real(x)

def concatenate(arrays, dim):
    return tf.stack(arrays, axis=dim)


def cdgmm(A, B):
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

    if A.shape[-len(B.shape):-1] != B.shape[:-1]:
        raise RuntimeError('The inputs are not compatible for multiplication.')

    return A * B
