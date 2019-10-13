import tensorflow as tf

BACKEND_NAME = 'tensorflow'
from collections import namedtuple

def complex_modulus(x):
    """Computes complex modulus.

        Parameters
        ----------
        x : tensor
            Input tensor whose complex modulus is to be calculated.

        Returns
        -------
        modulus : tensor
            Tensor the same size as input_array. modulus holds the
            result of the complex modulus.
             
    """
    modulus = tf.abs(x)
    return modulus


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
        output : tensor
            Tensor of the same size as input_array. It holds the output of
            the operation::

            $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$

            which is covariant to 3D translations and rotations.

    """
    if module is None:
        module = tf.zeros_like(x, tf.float32)
    else:
        module = module**2

    module += tf.abs(x)**2
    return tf.sqrt(module)


def _compute_standard_scattering_coefs(input_array, filter, J, subsample):
    """Computes convolution and downsampling.
    
        Computes the convolution of input_array with a lowpass filter phi_J
        and downsamples by a factor J.
    
        Parameters
        ----------
        input_array : tensor 
            Size (batchsize, M, N, O).
        filter : tensor
            Size (M, N, O).
        J : int
            Low pass scale of phi_J.
        subsample : function
            Subsampling function.
    
        Returns
        -------
        output : tensor 
            The result of input_array \\star phi_J downsampled by a factor J.
            
    """
    low_pass = filter[J]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    return subsample(convolved_input, J)



def _compute_local_scattering_coefs(input_array, filter, j, points):
    """Compute convolution and returns particular points.
    
        Computes the convolution of input_array with a lowpass filter phi_j+1
        and returns the value of the output at particular points.

        Parameters
        ----------
        input_array : tensor
            Size (batchsize, M, N, O).
        filter : tensor
            Size (M, N, O).
        j : int
            The lowpass scale j of phi_j.
        points : tensor
            Size (batchsize, number of points, 3).

        Returns
        -------
        output : tensor 
            Tensor of size (batchsize, number of points, 1) with the values
            of the lowpass filtered moduli at the points given.

    """
    local_coefs = tf.zeros(input_array.shape[0], points.shape[1])
    low_pass = filter[j+1]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    for i in range(input_array.shape[0]):
        for j in range(points.shape[1]):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            tf.assign(local_coefs[i, j], convolved_input[i, int(x), int(y), int(z)])
    return local_coefs


def subsample(input_array, j):
    """Downsamples.

        Parameters
        ----------
        input_array : tensor
            Input tensor.
        j : int
            Downsampling factor.

        Returns
        -------
        out : tensor
            Downsampled tensor. 
        
    """
    return input_array[..., ::2 ** j, ::2 ** j, ::2 ** j]


def compute_integrals(input_array, integral_powers):
    """Computes integrals.
        
        Computes integrals of the input_array to the given powers.

        Parameters
        ----------
        input_array : tensor
            Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
            dims.
        integral_powers : list
            List of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p
            norms).

        Returns
        -------
        integrals : tensor
            Tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms).

    """
    integrals = []
    for i_q, q in enumerate(integral_powers):
        integrals.append(tf.reduce_sum(tf.reshape(tf.pow(input_array, q), shape=(input_array.shape[0], -1)), axis=1))
    return tf.expand_dims(tf.stack(integrals, axis=-1), axis=-1)


def fft(x, direction='C2C', inverse=False):
    """FFT of a 3d signal.

        Example
        -------
        real = tf.random.uniform(128, 32, 32, 32)
        imag = tf.random.uniform(128, 32, 32, 32)
        
        x = tf.complex(real, imag)
        
        x_fft = fft(x)
        x_ifft = fft(x, inverse=True)

        x = fft(x_fft, inverse=True)
        x = fft(x_ifft, inverse=False)
        
        Parameters
        ----------
        input : tensor
            Complex input for the FFT.
        direction : string
        'C2R' for complex to real, 'C2C' for complex to complex.
        inverse : bool
            True for computing the inverse FFT.
            NB : If direction is equal to 'C2R', then an error is raised.
        
        Raises
        ------
        RuntimeError
            Raised in event we attempt to map from complex to real without
            inverse FFT.

        Returns
        -------
        output : tensor
            Result of FFT or IFFT.

    """
    x = tf.cast(x, tf.complex64)
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT')

    if direction == 'C2R':
        output = tf.real(tf.signal.ifft3d(x, name='irfft3d'))
    elif direction == 'C2C':
        if inverse:
            output = tf.signal.ifft3d(x, name='ifft3d')
        else:
            output = tf.signal.fft3d(x, name='fft3d')
    return tf.cast(output, tf.complex64)
   

def cdgmm3d(A, B, inplace=False):
    """Complex pointwise multiplication.
        
        Complex pointwise multiplication between (batched) tensor A and tensor B.
        
        Parameters
        ----------
        A : tensor
            Complex tensor.
        B : tensor
            Complex tensor of the same size as A.
        inplace : boolean, optional
            If set True, all the operations are performed inplace.
        
        Returns
        -------
        output : tensor 
            Tensor of the same size as A containing the result of the elementwise
            complex multiplication of A with B.

    """
    if B.ndim != 3:
        raise RuntimeError('The dimension of the second input must be 3.')

    #C = A * B
    import numpy as np
    Cr = tf.cast(tf.real(A)*np.real(B)-tf.imag(A)*np.imag(B),tf.complex64)
    Ci = tf.cast(tf.real(A)*np.imag(B)+tf.imag(A)*np.real(B),tf.complex64)

    return Cr+1.0j*Ci
   

def finalize(s_order_1, s_order_2, max_order):
    """Concatenate scattering of different orders.

        Parameters
        ----------
        s0 : tensor
            Tensor which contains the zeroth order scattering coefficents.
        s1 : tensor
            Tensor which contains the first order scattering coefficents.
        s2 : tensor
            Tensor which contains the second order scattering coefficents.
        
        Returns
        -------
        s : tensor
            Final output. Scattering transform.

    """
    s_order_1 = tf.stack([arr for arr in s_order_1], axis=2)
    if max_order == 2:
        s_order_2 = tf.stack([arr for arr in s_order_2], axis=2)
        return tf.squeeze(tf.concat([s_order_1, s_order_2], axis=1), axis=-1)
    else:
        return tf.squeeze(s_order_1, axis=-1)


def aggregate(x):
    """Aggregation of scattering coefficents.

        Parameters
        ----------
        x : list 
            List of tensors. 
    
        Returns
        -------
        out : tensor
            Stacked scattering coefficents.
    
    """
    return tf.stack([arr for arr in x], axis=1)

backend = namedtuple('backend', ['name', 'cdgmm3d', 'fft', 'finalize', 'modulus', 'modulus_rotation', 'subsample',\
                                 'compute_integrals', 'aggregate'])

backend.name = 'tensorflow'
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

