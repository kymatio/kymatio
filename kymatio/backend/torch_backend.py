from torch.autograd import Function
import torch

BACKEND_NAME = 'torch'

def input_checks(x):
    if x is None:
        raise TypeError('The input should be not empty.')

def complex_check(x):
    if not _is_complex(x):
        raise TypeError('The input should be complex (i.e. last dimension is 2).')

def real_check(x):
    if not _is_real(x):
        raise TypeError('The input should be real.')

def _is_complex(x):
    return x.dtype == torch.cfloat or x.dtype == torch.cdouble

def _is_real(x):
    return x.dtype == torch.float or x.dtype == torch.double

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

        output = x.abs()

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
        grad_input = x.mul(grad_output).div(output)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input

# shortcut for ModulusStable.apply
modulus = ModulusStable.apply

class Modulus():
    """This class implements a modulus transform for complex numbers.

        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)

        Parameters
        ---------
        x : tensor
            Complex torch tensor.

        Returns
        -------
        output : tensor
            A tensor with the same dimensions as x, such that output[..., 0]
            contains the complex modulus of x, while output[..., 1] = 0.
    """
    def __call__(self, x):
        complex_check(x)
        return x.abs()


def cdgmm(A, B):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2).
        B : tensor
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1).

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
    
    complex_check(A)

    if A.shape[-len(B.shape):] != B.shape:
        raise RuntimeError('The filters are not compatible for multiplication.')

    if (torch.view_as_real(A).dtype is not B.dtype) and (A.dtype is not B.dtype):
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

    return A*B

def concatenate(arrays, dim):
    return torch.stack(arrays, dim=dim)
