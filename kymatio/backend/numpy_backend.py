
   
class NumpyBackend:
    def __init__(self, name='numpy'):
        import numpy
        import scipy.fftpack
        self.np = numpy
        self.fft = scipy.fftpack
        self.name = name

    def input_checks(self, x):
        if x is None:
            raise TypeError('The input should be not empty.')
    
    def complex_check(self, x):
        if not self._is_complex(x):
            raise TypeError('The input should be complex.')
    
    def real_check(self, x):
        if not self._is_real(x):
            raise TypeError('The input should be real.')
    
    def _is_complex(self, x):
        return (x.dtype == self.np.complex64) or (x.dtype == self.np.complex128)
    
    def _is_real(self, x):
        return (x.dtype == self.np.float32) or (x.dtype == self.np.float64)
 
    def concatenate(self, arrays):
        return self.np.stack(arrays, axis=1)

    def modulus(self, x):
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
        return self.np.abs(x)
   
    def cdgmm(self, A, B):
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
        if not self._is_complex(A):
            raise TypeError('The first input must be complex.')
    
        if A.shape[-len(B.shape):] != B.shape[:]:
            raise RuntimeError('The inputs are not compatible for '
                               'multiplication.')
    
        if not self._is_complex(B) and not self._is_real(B):
            raise TypeError('The second input must be complex or real.')
    
        return A * B
