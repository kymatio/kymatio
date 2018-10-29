import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function


def pad1D(x, pad_left, pad_right, mode='constant', value=0.):
    """
    1D implementation of the padding function for torch tensors

    Parameters
    ----------
    x : tensor_like
        input tensor , 3D with time in the last axis.
    pad_left : int
        amount to add on the left of the tensor (at the beginning
        of the temporal axis)
    pad_right : int
        amount to add on the right of the tensor (at the end
        of the temporal axis)
    mode : string, optional
        Padding mode. Options include 'constant' and
        'reflect'. See the pytorch API for other options.
        Defaults to 'constant'
    value : float, optional
        If mode == 'constant', value to input
        within the padding. Defaults to 0.

    Returns
    -------
    res: the padded tensor
    """
    if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
        if mode == 'reflect':
            raise ValueError('Indefinite padding size (larger than tensor)')
    res = F.pad(x.unsqueeze(2),
                (pad_left, pad_right, 0, 0),
                mode=mode, value=value).squeeze(2)
    return res


class ModulusStable(Function):
    """
    This class implements a modulus transform for complex numbers which is
    stable with respect to very small inputs (z close to 0), avoiding
    returning nans in all cases.

    Usage
    -----
    modulus = ModulusStable.apply  # apply inherited from Function
    x_mod = modulus(x)

    Docstring for modulus
    ---------------------
    Function performing a modulus transform

    Parameters
    ---------
    x: input tensor, with last dimension = 2 for
        complex numbers
    p: optional, power of the norm (defaults to 2 for the complex modulus)
    dim: optional, dimension along which to reduce the norm (defaults to -1)
    keepdim: optional, whether to keep the dims after reduction or not
        (defaults to False)

    Returns
    -------
    output: a tensor of same size as the input tensor, except for the dimension
    of the complex numbers which is removed. This tensor is differentiable with
    respect to the input in a stable fashion (so diff modulus (0) = 0)
    """
    @staticmethod
    def forward(ctx, x, p=2, dim=-1, keepdim=False):
        """
        Forward pass of the modulus.
        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx: context tensors collected during the forward pass. They are
            automatically added by pytorch and should not be touched.
            They are then used for the backward pass.
        x: input tensor
        p: optional, power of the norm (defaults to 2 for the modulus)
        dim: optional, dimension along which to compute the norm
            (defaults to -1)
        keepdim: optional, whether to keep the dims after reduction or not
            (defaults to False)

        Returns
        -------
        output: torch tensor containing the modulus computing along the last
            axis, with this axis removed if keepdim is False
        """
        ctx.p = p
        ctx.dim = dim
        ctx.keepdim = False if keepdim is None else keepdim

        if dim is None:
            norm = x.norm(p)
            output = x.new((norm,))
        else:
            if keepdim is not None:
                output = x.norm(p, dim, keepdim=keepdim)
            else:
                output = x.norm(p, dim)
        ctx.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the modulus
        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx: context tensors collected during the forward pass, which allow
            to compute the backward pass
        grad_output: gradient with respect to the output tensor computed at
            the forward pass

        Returns
        -------
        grad_input: gradient with respect to the input, which should not
            contain any nan
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


def subsample_fourier(x, k):
    """
    Subsampling of a vector performed in the Fourier domain

    Subsampling in the temporal domain amounts to periodization
    in the Fourier domain, hence the formula.

    Parameters
    ----------
    x : tensor_like
        input tensor with at least 3 dimensions, the last
        corresponding to time (in the Fourier domain).
        The last dimension should be a power of 2 to avoid errors.
    k : int
        integer such that x is subsampled by 2**k

    Returns
    -------
    res : tensor_like
        tensor such that its fourier transform is the Fourier
        transform of a subsampled version of x, i.e. in
        FFT^{-1}(res)[t] = FFT^{-1}(x)[t * (2**k)]
    """
    N = x.shape[-2]
    res = x.view(x.shape[:-2] + (k, N // k, 2)).mean(dim=-3)
    return res


def pad(x, pad_left=0, pad_right=0, to_complex=True):
    """
    Padding which allows to simultaneously pad in a reflection fashion
    and map to complex if necessary

    Parameters
    ----------
    x : tensor_like
        input tensor, 3D with time in the last axis.
    pad_left : int, optional
        amount to add on the left of the tensor
        (at the beginning of the temporal axis). Defaults to 0
    pad_right : int, optional
        amount to add on the right of the tensor (at the end
        of the temporal axis). Defaults to 0
    to_complex : boolean, optional
        Whether to map the resulting padded tensor
        to a complex type (seen as a real number). Defaults to True

    Returns
    -------
    output : tensor_like
        A padded signal, possibly transformed into a 4D tensor
        with the last axis equal to 2 if to_complex is True (these
        dimensions should be interpreted as real and imaginary parts)
    """
    output = pad1D(x, pad_left, pad_right, mode='reflect')
    if to_complex:
        return torch.cat(
            [output.unsqueeze(-1),
             output.data.new(output.shape + (1,)).fill_(0.)],
            dim=-1)
    else:
        return output


def unpad(x, i0, i1):
    """
    Slices the input tensor at indices between i0 and i1 in the last axis

    Parameters
    ----------
    x : tensor_like
        input tensor (at least 1 axis)
    i0 : int
    i1: int

    Returns
    -------
    x[..., i0:i1]
    """
    return x[..., i0:i1]


def real(x):
    """
    Takes the real part of a 4D tensor x, where the last axis is interpreted
    as the real and imaginary parts.

    Parameters
    ----------
    x : tensor_like

    Returns
    -------
    x[..., 0], which is interpreted as the real part of x
    """
    return x[..., 0]


def modulus_complex(x):
    """
    Computes the modulus of x as a real tensor and returns a new complex tensor
    with imaginary part equal to 0.

    Parameters
    ----------
    x : tensor_like
        input tensor, should be a 4D tensor with the last axis of size 2

    Returns
    -------
    res : tensor_like
        a tensor with same dimensions as x, such that res[..., 0] contains
        the complex modulus of the input, and res[..., 1] = 0.
    """
    # take the stable modulus
    real = modulus(x)
    imag = real.data.new(real.shape).fill_(0.)
    res = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1)
    return res



def fft1d_c2c(x):
    """
        Computes the fft of a 1d signal.

        Input
        -----
        x : tensor
            the two final sizes must be (T, 2) where T is the time length of the signal
    """
    return torch.fft(x, signal_ndim = 1)


def ifft1d_c2c(x):
    """
        Computes the inverse fft of a 1d signal.

        Input
        -----
        x : tensor
            the two final sizes must be (T, 2) where T is the time length of the signal
    """
    return torch.ifft(x, signal_ndim = 1) * float(x.shape[-2])


def ifft1d_c2c_normed(x):
    """
        Computes the inverse normalized fft of a 1d signal.

        Input
        -----
        x : tensor
            the two final sizes must be (T, 2) where T is the time length of the signal
    """
    return torch.ifft(x, signal_ndim = 1)
