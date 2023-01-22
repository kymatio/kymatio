import torch
from torch.autograd import Function


class ModulusStable(Function):
    """Stable complex modulus

    This class implements a modulus transform for complex numbers which is
    stable with respect to very small inputs (z close to 0), avoiding
    returning nans in all cases.

    Usage
    -----
    modulus = ModulusStable.apply  # apply inherited from Function
    x_mod = modulus(x)

    Parameters
    ---------
    x : tensor
        The complex tensor (i.e., whose last dimension is two) whose modulus
        we want to compute.

    Returns
    -------
    output : tensor
        A tensor of same size as the input tensor, except for the last
        dimension, which is removed. This tensor is differentiable with respect
        to the input in a stable fashion (so gradent of the modulus at zero is
        zero).
    """
    @staticmethod
    def forward(ctx, x):
        """Forward pass of the modulus.

        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx : context object
            Collected during the forward pass. These are automatically added
            by PyTorch and should not be touched. They are then used for the
            backward pass.
        x : tensor
            The complex tensor whose modulus is to be computed.

        Returns
        -------
        output : tensor
            This contains the modulus computed along the last axis, with that
            axis removed.
        """
        ctx.p = 2
        ctx.dim = -1
        ctx.keepdim = False

        output = (x[...,0] * x[...,0] + x[...,1] * x[...,1]).sqrt()

        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the modulus

        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx : context object
            Collected during the forward pass. These are automatically added
            by PyTorch and should not be touched. They are then used for the
            backward pass.
        grad_output : tensor
            The gradient with respect to the output tensor computed at the
            forward pass.

        Returns
        -------
        grad_input : tensor
            The gradient with respect to the input.
        """
        x, output = ctx.saved_tensors

        if ctx.dim is not None and ctx.keepdim is False and x.dim() != 1:
            grad_output = grad_output.unsqueeze(ctx.dim)
            output = output.unsqueeze(ctx.dim)

        grad_input = x.mul(grad_output).div(output)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input


class TorchBackend:
    name = 'torch'

    @classmethod
    def input_checks(cls, x):
        if x is None:
            raise TypeError('The input should be not empty.')

        cls.contiguous_check(x)

    @classmethod
    def complex_check(cls, x):
        if not cls._is_complex(x):
            raise TypeError('The input should be complex (i.e. last dimension is 2).')

    @classmethod
    def real_check(cls, x):
        if not cls._is_real(x):
            raise TypeError('The input should be real.')

    @classmethod
    def complex_contiguous_check(cls, x):
        cls.complex_check(x)
        cls.contiguous_check(x)

    @staticmethod
    def contiguous_check(x):
        if not x.is_contiguous():
            raise RuntimeError('Tensors must be contiguous.')

    @staticmethod
    def _is_complex(x):
        return x.shape[-1] == 2

    @staticmethod
    def _is_real(x):
        return x.shape[-1] == 1

    @classmethod
    def modulus(cls, x):
        cls.complex_contiguous_check(x)
        norm = ModulusStable.apply(x)[..., None]
        return norm

    @staticmethod
    def stack(arrays, dim=2):
        return torch.stack(arrays, dim=dim)

    @classmethod
    def cdgmm(cls, A, B):
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
        if not cls._is_real(B):
            cls.complex_contiguous_check(B)
        else:
            cls.contiguous_check(B)

        cls.complex_contiguous_check(A)

        if A.shape[-len(B.shape):-1] != B.shape[:-1]:
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

        if cls._is_real(B):
            return A * B
        else:
            C = A.new(A.shape)

            A_r = A[..., 0].view(-1, B.nelement() // 2)
            A_i = A[..., 1].view(-1, B.nelement() // 2)

            B_r = B[..., 0].view(-1).unsqueeze(0).expand_as(A_r)
            B_i = B[..., 1].view(-1).unsqueeze(0).expand_as(A_i)

            C[..., 0].view(-1, B.nelement() // 2)[:] = A_r * B_r - A_i * B_i
            C[..., 1].view(-1, B.nelement() // 2)[:] = A_r * B_i + A_i * B_r

            return C

    @staticmethod
    def reshape_input(x, signal_shape):
        return x.reshape((-1, 1) + signal_shape)

    @staticmethod
    def reshape_output(S, batch_shape, n_kept_dims):
        new_shape = batch_shape + S.shape[-n_kept_dims:]
        return S.reshape(new_shape)

    @staticmethod
    def shape(x):
        return x.shape
