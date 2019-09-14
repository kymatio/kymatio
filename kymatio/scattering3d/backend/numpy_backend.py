import numpy as np
import warnings

BACKEND_NAME = 'numpy'
from collections import namedtuple

def complex_modulus(input_array):
    return np.abs(input_array)



def fft(x, direction='C2C', inverse=False):
    """
        fft of a 3d signal

        Example
        -------
        x = torch.randn(128, 32, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        inverse : bool
            True for computing the inverse FFT.
.
    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if direction == 'C2R':
        output = np.real(np.fft.ifftn(x, axes=(-3,-2,-1)))*x.shape[-1]*x.shape[-2]
    elif direction == 'C2C':
        if inverse:
            output = np.fft.ifftn(x, axes=(-3,-2,-1))*x.shape[-1]*x.shape[-2]
        else:
            output = np.fft.fftn(x, axes=(-3,-2,-1))
    return output


def cdgmm3d(A, B, inplace=False):
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

    if B.ndim != 3:
        raise RuntimeError('The dimension of the second input must be 3.')

    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B

def finalize(s_order_1, s_order_2, max_order):
    s_order_1 =   np.concatenate(s_order_1, axis=1)
    if max_order == 2:
        s_order_2 = np.concatenate(s_order_2, axis=1)
        return np.concatenate([s_order_1, s_order_2], axis=1)
    else:
        return s_order_1



def modulus_rotation(x, module):
    """
    Computes the convolution with a set of solid harmonics of scale j and
    degree l and returns the square root of their squared sum over m

    Parameters
    ----------
    input_array : tensor
        size (batchsize, M, N, O, 2)
    l : int
        solid harmonic degree l

    j : int
        solid harmonic scale j

    Returns
    -------

    output : torch tensor
        tensor of the same size as input_array. It holds the output of
        the operation::

        $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$

        which is covariant to 3D translations and rotations

    """
    if module is None:
        module = np.zeros_like(x)
    else:
        module = module **2
    module += np.abs(x)**2
    return np.sqrt(module)



def _compute_standard_scattering_coefs(input_array, low_pass, J, subsample):
    """
    Computes the convolution of input_array with a lowpass filter phi_J
    and downsamples by a factor J.

    Parameters
    ----------
    input_array: torch tensor of size (batchsize, M, N, O, 2)

    Returns
    -------
    output: the result of input_array \\star phi_J downsampled by a factor J

    """
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    return subsample(convolved_input, J)



def _compute_local_scattering_coefs(input_array, low_pass, points):
    """
    Computes the convolution of input_array with a lowpass filter phi_j and
    and returns the value of the output at particular points

    Parameters
    ----------
    input_array: torch tensor
        size (batchsize, M, N, O, 2)
    points: torch tensor
        size (batchsize, number of points, 3)
    j: int
        the lowpass scale j of phi_j

    Returns
    -------
    output: torch tensor of size (batchsize, number of points, 1) with
            the values of the lowpass filtered moduli at the points given.

    """
    local_coefs = np.zeros((input_array.shape[0], points.shape[1]), dtype=np.complex64)
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    for i in range(input_array.shape[0]):
        for j in range(points[i].shape[0]):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs



def subsample(input_array, j):
    return input_array[..., ::2 ** j, ::2 ** j, ::2 ** j, :].contiguous()



def compute_integrals(input_array, integral_powers):
    """
        Computes integrals of the input_array to the given powers.

        Parameters
        ----------
        input_array: torch tensor
            size (B, M, N, O), B batch_size, M, N, O spatial dims

        integral_powers: list
            list of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p norms)

        Returns
        -------
        integrals: torch tensor
            tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms)

    """
    integrals = np.zeros((input_array.shape[0], len(integral_powers)),dtype=np.complex64)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = (input_array ** q).reshape((
                                        input_array.shape[0], -1)).sum(axis=1)
    return integrals


def aggregate(x):
    return np.concatenate([arr for arr in x], axis=1)

backend = namedtuple('backend', ['name', 'cdgmm3d', 'fft', 'finalize', 'modulus', 'modulus_rotation', 'subsample',
                                 'compute_integrals', 'aggregate'])

backend.name = 'numpy'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.aggregate = aggregate
backend.finalize = finalize
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.subsample = subsample
backend.compute_integrals = compute_integrals
backend._compute_standard_scattering_coefs = _compute_standard_scattering_coefs
backend._compute_local_scattering_coefs = _compute_local_scattering_coefs

