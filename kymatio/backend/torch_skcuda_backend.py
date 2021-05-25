import torch


class TorchSkcudaBackend:
    name = 'torch_skcuda'

    @staticmethod
    def _is_complex(x):
        return torch.is_complex(x)

    @staticmethod
    def _is_real(x):
        return 'float' in str(x.dtype)

    @classmethod
    def cdgmm(cls, A, B):
        """Complex pointwise multiplication.

            Complex pointwise multiplication between (batched) tensor A and tensor
            B.

            Parameters
            ----------
            A : tensor
                A is a complex tensor of size (B, C, M, N, 2).
            B : tensor
                B is a complex tensor of size (M, N, 2) or real tensor of (M, N,
                1).
            inplace : boolean, optional
                If set to True, all the operations are performed in place.

            Raises
            ------
            RuntimeError
                In the event that the filter B is not a 3-tensor with a last
                dimension of size 1 or 2, or A and B are not compatible for
                multiplication, or if A or B are not contiguous.
            TypeError
                In the event that A is not complex, or B does not have a final
                dimension of 1 or 2, or A and B are not of the same dtype, or
                if A or B are not cuda tensors, or if A and B are not on the same
                device.

            Returns
            -------
            C : tensor
                Output tensor of size (B, C, M, N, 2) such that:
                C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

        """
        if not cls._is_complex(A):
            raise TypeError('The input should be complex (got %s).' % A.dtype)

        if not (cls._is_complex(B) or cls._is_real(B)):
            raise TypeError('The filter should be complex or real, indicated by a '
                            'last dimension of size 2 or 1, respectively.')

        if A.shape[-B.ndim:-1] != B.shape[:-1]:
            raise RuntimeError('The filters are not compatible for multiplication.')

        if not A.is_cuda or not B.is_cuda:
            raise TypeError('Input and filter must be CUDA tensors.')

        if A.device.index != B.device.index:
            raise TypeError('Input and filter must be on the same GPU.')

        return A * B
