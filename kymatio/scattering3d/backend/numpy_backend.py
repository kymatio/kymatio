import numpy as np
import warnings

BACKEND_NAME = 'numpy'
from collections import namedtuple

def complex_modulus(x):
    """Compute the complex modulus.

        Computes the modulus of x and stores the result in a real numpy array.
        
        Parameters
        ----------
        x : numpy array
            A complex numpy array.
    
        Returns
        -------
        norm : numpy array
            A real numpy array with the same dimensions as x. Real part
            contains complex modulus of x.
    
    """
    return np.abs(x)



def modulus_rotation(x, module):
    """Used for computing rotation invariant scattering transform coefficents.
 
        Parameters
        ----------
        x : tensor
            Size (batchsize, M, N, O).
        module : tensor
            Tensor that holds the overall sum.
   
        Returns
        -------
        output : numpy array
            Numpy array of the same size as input_array. It holds the output of
            the operation::
    
            $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$
    
            which is covariant to 3D translations and rotations.
    
   """
    if module is None:
        module = np.zeros_like(x)
    else:
        module = module **2
    module += np.abs(x)**2
    return np.sqrt(module)



def _compute_standard_scattering_coefs(input_array, filter, J, subsample):
    """Computes convolution and downsamples.
    
        Computes the convolution of input_array with a lowpass filter phi_J
        and downsamples by a factor J.
    
        Parameters
        ----------
        input_array : numpy array 
            Size (batchsize, M, N, O).
        filter : numpy array
            Size (M, N, O).
        J : int
            Low pass scale of phi_J.
        subsample : function
            Subsampling function.
    
        Returns
        -------
        output : numpy array 
            The result of input_array \\star phi_J downsampled by a factor J.
    
    """
    low_pass = filter[J]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    return subsample(convolved_input, J)


def _compute_local_scattering_coefs(input_array, filter, j, points):
    """Compute convolution and returns particular points.

        Computes the convolution of input_array with a lowpass filter phi_j and
        and returns the value of the output at particular points.

        Parameters
        ----------
        input_array : numpy array
            Size (batchsize, M, N, O, 2).
        filter : numpy array
            Size (M, N, O, 2)
        j : int
            The lowpass scale j of phi_j
        points : numpy array
            Size (batchsize, number of points, 3)

        Returns
        -------
        output : numpy array
            Numpy array of size (batchsize, number of points, 1) with the values
            of the lowpass filtered moduli at the points given.
            
    """
    local_coefs = np.zeros((input_array.shape[0], points.shape[1]), dtype=np.complex64)
    low_pass = filter[j+1]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    for i in range(input_array.shape[0]):
        for j in range(points[i].shape[0]):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs



def subsample(input_array, j):
    """Downsamples.

        Parameters
        ----------
        input_array : numpy array
            Input numpy array.
        j : int
            Downsampling factor.

        Returns
        -------
        out : numpy array
            Downsampled numpy array. 
        
    """
    return np.ascontiguousarray(input_array[..., ::2 ** j, ::2 ** j, ::2 ** j])


def compute_integrals(input_array, integral_powers):
    """Computes integrals.

        Computes integrals of the input_array to the given powers.

        Parameters
        ----------
        input_array: numpy array
            Size (B, M, N, O), B is batch_size, M, N, O are spatial dims.

        integral_powers: list
            List of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p
            norms).

        Returns
        -------
        integrals: numpy array
            Numpy array of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms).

    """
    integrals = np.zeros((input_array.shape[0], len(integral_powers)),dtype=np.complex64)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = (input_array ** q).reshape((
                                        input_array.shape[0], -1)).sum(axis=1)
    return integrals


def fft(x, direction='C2C', inverse=False):
    """FFT of a 3d signal.

        Example
        -------
        x = numpy.random.randn(128, 32, 32, 32, 2).view(numpy.complex64)
        x_fft = fft(x)
        x_ifft = fft(x, inverse=True)

        Parameters
        ----------
        input : numpy array
            Complex input for the FFT.
        inverse : bool
            True for computing the inverse FFT.

        Raises
        ------
        RuntimeError
            Raised in event we attempt to map from complex to real without
            inverse FFT.

        Returns
        -------
        output : numpy array
            Result of FFT or IFFT.

    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if direction == 'C2R':
        output = np.real(np.fft.ifftn(x, axes=(-3,-2,-1)))
    elif direction == 'C2C':
        if inverse:
            output = np.fft.ifftn(x, axes=(-3,-2,-1))
        else:
            output = np.fft.fftn(x, axes=(-3,-2,-1))
    return output


def cdgmm3d(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) numpy array A and
        numpy array B.

        Parameters
        ----------
        A : numpy array
            A is a complex numpy array of size (B, C, M, N).
        B : numpy array
            B is a complex or real numpy array of size (M, N).
        inplace : boolean, optional
            If set to True, all the operations are performed inplace.

        Raises
        ------
        RuntimeError
            Raised in event B is not three dimensional.
        
        Returns
        -------
        C : numpy array
            Output numpy array of size (B, C, M, N) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

    """
    if B.ndim != 3:
        raise RuntimeError('The dimension of the second input must be 3.')

    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B


def finalize(s_order_1, s_order_2, max_order):
    """Concatenate scattering of different orders.
    
        Parameters
        ----------
        s0 : numpy array
            numpy array which contains the zeroth order scattering coefficents.
        s1 : numpy array
            numpy array which contains the first order scattering coefficents.
        s2 : numpy array
            numpy array which contains the second order scattering coefficents.
        
        Returns
        -------
        s : numpy array
            Final output. Scattering transform.
    
        """
    s_order_1 = np.concatenate([np.expand_dims(arr, 2) for arr in s_order_1], axis=2)
    if max_order == 2:
        s_order_2 = np.concatenate([np.expand_dims(arr, 2) for arr in s_order_2], axis=2)
        return np.concatenate([s_order_1, s_order_2], axis=1)
    else:
        return s_order_1


def aggregate(x):
    """Aggregation of scattering coefficents.

        Parameters
        ----------
        x : list 
            List of numpy arrays. 

        Returns
        -------
        out : numpy array
            Stacked scattering coefficents.

    """
    return np.concatenate([np.expand_dims(arr, 1) for arr in x], axis=1)

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

