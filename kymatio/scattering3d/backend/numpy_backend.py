import numpy as np
from collections import namedtuple
from scipy.fftpack import fftn, ifftn


BACKEND_NAME = 'numpy'


def _iscomplex(x):
    return x.dtype == np.complex64 or x.dtype == np.complex128


def complex_modulus(input_array):
    """Computes complex modulus.

        Parameters
        ----------
        input_array : tensor
            Input tensor whose complex modulus is to be calculated.
        Returns
        -------
        modulus : tensor
            Tensor the same size as input_array. modulus[..., 0] holds the
            result of the complex modulus, modulus[..., 1] = 0.

    """

    return np.abs(input_array)


def modulus_rotation(x, module=None):
    """Used for computing rotation invariant scattering transform coefficents.

        Parameters
        ----------
        x : tensor
            Size (batchsize, M, N, O, 2).
        module : tensor
            Tensor that holds the overall sum. If none, initializes the tensor
            to zero (default).
        Returns
        -------
        output : torch tensor
            Tensor of the same size as input_array. It holds the output of
            the operation::
            $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$
            which is covariant to 3D translations and rotations.
    """
    if module is None:
        module = np.zeros_like(x)
    else:
        module = module ** 2
    module += np.abs(x) ** 2
    return np.sqrt(module)


def compute_integrals(input_array, integral_powers):
    """Computes integrals.

        Computes integrals of the input_array to the given powers.
        Parameters
        ----------
        input_array : torch tensor
            Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
            dims.
        integral_powers : list
            List of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p
            norms).
        Returns
        -------
        integrals : torch tensor
            Tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms).
    """
    integrals = np.zeros((input_array.shape[0], len(integral_powers)),
            dtype=np.complex64)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = (input_array ** q).reshape((input_array.shape[0], -1)).sum(axis=1)
    return integrals


def fft(x, direction='C2C', inverse=False):
    """
        Interface with torch FFT routines for 2D signals.

        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then an error is raised.

    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if direction == 'C2R':
        output = np.real(ifftn(x, axes=(-3, -2, -1)))
    elif direction == 'C2C':
        if inverse:
            output = ifftn(x, axes=(-3, -2, -1))
        else:
            output = fftn(x, axes=(-3, -2, -1))

    return output


def cdgmm3d(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : torch tensor
            Complex torch tensor.
        B : torch tensor
            Complex of the same size as A.
        inplace : boolean, optional
            If set True, all the operations are performed inplace.

        Raises
        ------
        RuntimeError
            In the event that the tensors are not compatibile for multiplication
            (i.e. the final four dimensions of A do not match with the dimensions
            of B), or in the event that B is not complex, or in the event that the
            type of A and B are not the same.
        TypeError
            In the event that x is not complex i.e. does not have a final dimension
            of 2, or in the event that both tensors are not on the same device.

        Returns
        -------
        output : torch tensor
            Torch tensor of the same size as A containing the result of the
            elementwise complex multiplication of A with B.
    """

    if A.shape[-3:] != B.shape[-3:]:
        raise RuntimeError('The tensors are not compatible for multiplication.')

    if not _iscomplex(A) or not _iscomplex(B):
        raise TypeError('The input, filter and output should be complex.')

    if B.ndim != 3:
        raise RuntimeError('The second tensor must be simply a complex array.')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type.')


    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B


def concatenate(arrays, L):
    S = np.stack(arrays, axis=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    return S


backend = namedtuple('backend',
                     ['name',
                      'cdgmm3d',
                      'fft',
                      'modulus',
                      'modulus_rotation',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'numpy'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.concatenate = concatenate
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.compute_integrals = compute_integrals
