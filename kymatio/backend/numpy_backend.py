import numpy as np

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

def _iscomplex(x):
    return x.dtype == np.complex64 or x.dtype == np.complex128


def _isreal(x):
    return x.dtype == np.float32 or x.dtype == np.float64

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
    if A.shape[-len(B.shape):-1] != B.shape[:-1]:
        raise RuntimeError('The inputs are not compatible for '
                           'multiplication.')

    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B