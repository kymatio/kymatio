import torch
from torch.autograd import Function


from .base_backend import BaseBackend, backend_types, backend_basic_math, backend_array



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


    # Add for new backend enhancement effort, copied and pasted from 
    # https://github.com/tensorly/tensorly/blob/main/tensorly/backend/pytorch_backend.py


    @staticmethod
    def shape(tensor):
        return tuple(tensor.shape)

    @staticmethod
    def ndim(tensor):
        return tensor.dim()

    @staticmethod
    def arange(start, stop=None, step=1.0, *args, **kwargs):
        if stop is None:
            return torch.arange(
                start=0.0, end=float(start), step=float(step), *args, **kwargs
            )
        else:
            return torch.arange(float(start), float(stop), float(step), *args, **kwargs)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        if inplace:
            return torch.clip(tensor, a_min, a_max, out=tensor)
        else:
            return torch.clip(tensor, a_min, a_max)

    ## This one seems wrong
    # @staticmethod
    # def all(tensor):
    #     return torch.sum(tensor != 0)

    def transpose(self, tensor, axes=None):
        axes = axes or list(range(self.ndim(tensor)))[::-1]
        return tensor.permute(*axes)

    @staticmethod
    def copy(tensor):
        return tensor.clone()

    @staticmethod
    def norm(tensor, order=None, axis=None):
        # pytorch does not accept `None` for any keyword arguments. additionally,
        # pytorch doesn't seems to support keyword arguments in the first place
        kwds = {}
        if axis is not None:
            kwds["dim"] = axis
        if order and order != "inf":
            kwds["p"] = order

        if order == "inf":
            res = torch.max(torch.abs(tensor), **kwds)
            if axis is not None:
                return res[0]  # ignore indices output
            return res
        return torch.norm(tensor, **kwds)

    @staticmethod
    def dot(a, b):
        if a.ndim > 2 and b.ndim > 2:
            return torch.tensordot(a, b, dims=([-1], [-2]))
        if not a.ndim or not b.ndim:
            return a * b
        return torch.matmul(a, b)

    @staticmethod
    def tensordot(a, b, axes=2, **kwargs):
        return torch.tensordot(a, b, dims=axes, **kwargs)

    @staticmethod
    def mean(tensor, axis=None):
        if axis is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=axis)

    @staticmethod
    def sum(tensor, axis=None, keepdims=False):
        if axis is None:
            axis = tuple(range(tensor.ndim))
        return torch.sum(tensor, dim=axis, keepdim=keepdims)

    @staticmethod
    def max(tensor, axis=None):
        if axis is None:
            return torch.max(tensor)
        else:
            return torch.max(tensor, dim=axis)[0]

    @staticmethod
    def flip(tensor, axis=None):
        if isinstance(axis, int):
            axis = [axis]

        if axis is None:
            return torch.flip(tensor, dims=[i for i in range(tensor.ndim)])
        else:
            return torch.flip(tensor, dims=axis)

    # @staticmethod
    # def concatenate(tensors, axis=0):
    #     return torch.cat(tensors, dim=axis)

    @staticmethod
    def argmin(input, axis=None):
        return torch.argmin(input, dim=axis)

    @staticmethod
    def argsort(input, axis=None):
        return torch.argsort(input, dim=axis)

    @staticmethod
    def argmax(input, axis=None):
        return torch.argmax(input, dim=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def diag(tensor, k=0):
        return torch.diag(tensor, diagonal=k)

    @staticmethod
    def sort(tensor, axis):
        if axis is None:
            tensor = tensor.flatten()
            axis = -1

        return torch.sort(tensor, dim=axis).values



# Now add all the functions that are unchanged
# by just adding them in programmatically

remaining_funcs = [
        "nan",
        "is_tensor",
        "trace",
        "conj",
        "finfo",
        "log2",
        "digamma",
    ]

for func_name in backend_types + backend_basic_math + backend_array + remaining_funcs:
    if not hasattr(TorchBackend, func_name):
        setattr(TorchBackend, func_name, getattr(torch, func_name))
    else:
        raise ValueError(f"Function {func_name} already exists in TorchBackend")
    
