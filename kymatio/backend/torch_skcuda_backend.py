import torch
from skcuda import cublas


class TorchSkcudaBackend:
    name = 'torch_skcuda'

    @staticmethod
    def _is_complex(x):
        return x.shape[-1] == 2

    @staticmethod
    def _is_real(x):
        return x.shape[-1] == 1

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
            raise TypeError('The input should be complex (i.e. last dimension is 2).')

        if not cls._is_complex(B) and not cls._is_real(B):
            raise TypeError('The filter should be complex or real, indicated by a '
                            'last dimension of size 2 or 1, respectively.')

        if A.shape[-len(B.shape):-1] != B.shape[:-1]:
            raise RuntimeError('The filters are not compatible for multiplication.')

        if A.dtype is not B.dtype:
            raise TypeError('Input and filter must be of the same dtype.')

        if not A.is_cuda or not B.is_cuda:
            raise TypeError('Input and filter must be CUDA tensors.')

        if A.device.index != B.device.index:
            raise TypeError('Input and filter must be on the same GPU.')

        if cls._is_real(B):
            return A * B
        else:
            if not A.is_contiguous() or not B.is_contiguous():
                raise RuntimeError('Tensors must be contiguous.')

            C = torch.empty_like(A)
            m, n = B.nelement() // 2, A.nelement() // B.nelement()
            lda = m
            ldc = m
            incx = 1
            handle = torch.cuda.current_blas_handle()
            stream = torch.cuda.current_stream()._as_parameter_
            cublas.cublasSetStream(handle, stream)
            cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
            return C
