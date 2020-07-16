from .numpy_backend import NumpyBackend
import tensorflow as tf


class TensorFlowBackend(NumpyBackend):
    def __init__(self):
        super(TensorFlowBackend, self).__init__(name='tensorflow')

    def concatenate(self, arrays):
        return tf.stack(arrays, axis=1)

    def modulus(self, x):
        norm = tf.abs(x)
        
        return norm
    
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
            raise RuntimeError('The inputs are not compatible for multiplication.')
    
        if not self._is_complex(B) and not self._is_real(B):
            raise TypeError('The second input must be complex or real.')
    
        return A * B
