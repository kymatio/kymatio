import numpy as np

def input_checks(x):
    if x is None:
        raise TypeError('The input should be not empty.')


def modulus(x):
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
    return np.abs(x)


def _is_complex(x):
    return (x.dtype == np.complex64) or (x.dtype == np.complex128)


def _is_real(x):
    return (x.dtype == np.float32) or (x.dtype == np.float64)


def cdgmm(A, B, inplace=False):
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

    if not _is_complex(A):
        raise TypeError('The first input must be complex.')

    if A.shape[-len(B.shape):] != B.shape[:]:
        raise RuntimeError('The inputs are not compatible for '
                           'multiplication.')

    if not _is_complex(B) and not _is_real(B):
        raise TypeError('The second input must be complex or real.')

    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B


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
    return np.real(x)
