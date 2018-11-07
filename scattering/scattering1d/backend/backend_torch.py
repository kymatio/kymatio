# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function

NAME = 'torch'

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

        output = (x[...,0]*x[...,0] + x[...,1]*x[...,1]).sqrt()

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

        if ctx.p == 2:
            grad_input = x.mul(grad_output).div(output)
        else:
            input_pow = x.abs().pow(ctx.p - 2)
            output_pow = output.pow(ctx.p - 1)
            grad_input = x.mul(input_pow).mul(grad_output).div(output_pow)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input, None, None, None

# shortcut for ModulusStable.apply
modulus = ModulusStable.apply

def modulus_complex(x):
    """Compute the complex modulus

    Computes the modulus of x and stores the result in a complex tensor of the
    same size, with the real part equal to the modulus and the imaginary part
    equal to zero.

    Parameters
    ----------
    x : tensor
        A complex tensor (that is, whose last dimension is equal to 2).

    Returns
    -------
    res : tensor
        A tensor with the same dimensions as x, such that res[..., 0] contains
        the complex modulus of x, while res[..., 1] = 0.
    """
    norm = modulus(x)

    res = torch.zeros_like(x)
    res[...,0] = norm

    return res

def subsample_fourier(x, k):
    """Subsampling in the Fourier domain

    Subsampling in the temporal domain amounts to periodization in the Fourier
    domain, so the input is periodized according to the subsampling factor.

    Parameters
    ----------
    x : tensor
        Input tensor with at least 3 dimensions, where the next to last
        corresponds to the frequency index in the standard PyTorch FFT
        ordering. The length of this dimension should be a power of 2 to
        avoid errors. The last dimension should represent the real and
        imaginary parts of the Fourier transform.
    k : int
        The subsampling factor.

    Returns
    -------
    res : tensor
        The input tensor periodized along the next to last axis to yield a
        tensor of size x.shape[-2] // k along that dimension.
    """
    N = x.shape[-2]
    res = x.view(x.shape[:-2] + (k, N // k, 2)).mean(dim=-3)
    return res

def pad_1d(x, pad_left, pad_right, mode='constant', value=0.):
    """Pad real 1D tensors

    1D implementation of the padding function for real PyTorch tensors.

    Parameters
    ----------
    x : tensor
        Three-dimensional input tensor with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the tensor (at the beginning of the
        temporal axis).
    pad_right : int
        amount to add on the right of the tensor (at the end of the temporal
        axis).
    mode : string, optional
        Padding mode. Options include 'constant' and 'reflect'. See the
        PyTorch API for other options.  Defaults to 'constant'.
    value : float, optional
        If mode == 'constant', value to input within the padding. Defaults to
        0.

    Returns
    -------
    res : tensor
        The tensor passed along the third dimension.
    """
    if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
        if mode == 'reflect':
            raise ValueError('Indefinite padding size (larger than tensor).')
    res = F.pad(x.unsqueeze(2),
                (pad_left, pad_right, 0, 0),
                mode=mode, value=value).squeeze(2)
    return res

def pad(x, pad_left=0, pad_right=0, to_complex=True):
    """Pad real 1D tensors and map to complex

    Padding which allows to simultaneously pad in a reflection fashion and map
    to complex if necessary.

    Parameters
    ----------
    x : tensor
        Three-dimensional input tensor with the third axis being the one to
        be padded.
    pad_left : int
        Amount to add on the left of the tensor (at the beginning of the
        temporal axis).
    pad_right : int
        amount to add on the right of the tensor (at the end of the temporal
        axis).
    to_complex : boolean, optional
        Whether to map the resulting padded tensor to a complex type (seen
        as a real number). Defaults to True.

    Returns
    -------
    output : tensor
        A padded signal, possibly transformed into a four-dimensional tensor
        with the last axis of size 2 if to_complex is True (this axis
        corresponds to the real and imaginary parts).
    """
    output = pad_1d(x, pad_left, pad_right, mode='reflect')
    if to_complex:
        output = torch.stack((output, torch.zeros_like(output)), dim=-1)
    return output

def unpad(x, i0, i1):
    """Unpad real 1D tensor

    Slices the input tensor at indices between i0 and i1 along the last axis.

    Parameters
    ----------
    x : tensor
        Input tensor with least one axis.
    i0 : int
        Start of original signal before padding.
    i1 : int
        End of original signal before padding.

    Returns
    -------
    x_unpadded : tensor
        The tensor x[..., i0:i1].
    """
    return x[..., i0:i1]

def real(x):
    """Real part of complex tensor

    Takes the real part of a complex tensor, where the last axis corresponds
    to the real and imaginary parts.

    Parameters
    ----------
    x : tensor
        A complex tensor (that is, whose last dimension is equal to 2).

    Returns
    -------
    x_real : tensor
        The tensor x[..., 0] which is interpreted as the real part of x.
    """
    return x[..., 0]

def fft1d_c2c(x):
    """Compute the 1D FFT of a complex signal

    Input
    -----
    x : tensor
        A tensor of size (..., T, 2), where x[..., 0] is the real part and
        x[..., 1] is the imaginary part.

    Returns
    -------
    x_f : tensor
        A tensor of the same size as x containing its Fourier transform in the
        standard PyTorch FFT ordering.
    """
    return torch.fft(x, signal_ndim=1)

def ifft1d_c2c(x):
    """Compute the normalized 1D inverse FFT of a complex signal

    Input
    -----
    x_f : tensor
        A tensor of size (..., T, 2), where x_f[..., 0] is the real part and
        x[..., 1] is the imaginary part. The frequencies are assumed to be in
        the standard PyTorch FFT ordering.

    Returns
    -------
    x : tensor
        A tensor of the same size of x_f containing the normalized inverse
        Fourier transform of x_f.
    """
    return torch.ifft(x, signal_ndim=1)
