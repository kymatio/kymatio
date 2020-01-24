BACKEND_NAME = 'torch'

def _iscomplex(x):
    return x.shape[-1] == 2

def _isreal(x):
    return x.shape[-1] == 1

def cdgmm(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2).
        B : tensor
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1).
        inplace : boolean, optional
            If set to True, all the operations are performed in place.

        Raises
        ------
        RuntimeError
            In the event that the filter B is not a 3-tensor with a last
            dimension of size 1 or 2, or A and B are not compatible for
            multiplication.
        TypeError
            In the event that A is not complex, or B does not have a final
            dimension of 1 or 2, or A and B are not of the same dtype, or if
            A and B are not on the same device.

        Returns
        -------
        C : tensor
            Output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

    """
    if not _iscomplex(A):
        raise TypeError('The input must be complex, indicated by a last '
                        'dimension of size 2.')

    if not _iscomplex(B) and not _isreal(B):
        raise TypeError('The filter must be complex or real, indicated by a '
                        'last dimension of size 2 or 1, respectively.')

    if A.shape[:-1] != B.shape[:-1]:
        raise RuntimeError('The filters are not compatible for multiplication.')

    if A.dtype is not B.dtype:
        raise TypeError('Input and filter must be of the same dtype.')

    if B.device.type == 'cuda':
        if A.device.type == 'cuda':
            if A.device.index != B.device.index:
                raise TypeError('Input and filter must be on the same GPU.')
        else:
            raise TypeError('Input must be on GPU.')

    if B.device.type == 'cpu':
        if A.device.type == 'cuda':
            raise TypeError('Input must be on CPU.')

    if not A.is_contiguous() or not B.is_contiguous():
        raise RuntimeError('Tensors must be contiguous.')

    if _isreal(B):
        if inplace:
            return A.mul_(B)
        else:
            return A * B
    else:
        C = A.new(A.shape)

        A_r = A[..., 0].view(-1, A.shape[-2]*A.shape[-3])
        A_i = A[..., 1].view(-1, A.shape[-2]*A.shape[-3])

        B_r = B[...,0].view(B.shape[-2]*B.shape[-3]).unsqueeze(0).expand_as(A_i)
        B_i = B[..., 1].view(B.shape[-2]*B.shape[-3]).unsqueeze(0).expand_as(A_r)

        C[..., 0].view(-1, C.shape[-2]*C.shape[-3])[:] = A_r * B_r - A_i * B_i
        C[..., 1].view(-1, C.shape[-2]*C.shape[-3])[:] = A_r * B_i + A_i * B_r

        return C if not inplace else A.copy_(C)