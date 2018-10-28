import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function


def pad1D(x, pad_left, pad_right, mode='constant', value=0.):
    """
    1D implementation of the padding function for torch tensors
    or variables.

    Parameters
    ----------
    x : tensor_like
        input tensor (or variable), 3D with time in the last axis.
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
    res: the padded tensor (or variable)
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
    x: input tensor (embedded in a Variable), with last dimension = 2 for
        complex numbers
    p: optional, power of the norm (defaults to 2 for the complex modulus)
    dim: optional, dimension along which to reduce the norm (defaults to -1)
    keepdim: optional, whether to keep the dims after reduction or not
        (defaults to False)

    Returns
    -------
    output: a tensor embedded in a Variable of same size as the input
        tensor, except for the dimension of the complex numbers which is
        removed. This variable is differentiable with respect to the input
        in a stable fashion (so diff modulus (0) = 0)
    """
    @staticmethod
    def forward(ctx, x, p=2, dim=-1, keepdim=False):
        """
        Forward pass of the modulus.
        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx: context variables collected during the forward pass. They are
            automatically added by pytorch and should not be touched.
            They are then used for the backward pass.
        x: input tensor (already de-variabled).
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
        ctx: context variables collected during the forward pass, which allow
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


def ceiling_strict(s):
    """
    String ceiling of an input.
    """
    c = np.ceil(s)
    if c == s:
        return int(s + 1)
    else:
        return int(c)


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
        input tensor (or variable), 3D with time in the last axis.
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
             Variable(output.data.new(output.shape + (1,)).fill_(0.))],
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
    imag = Variable(real.data.new(real.shape).fill_(0.))
    res = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1)
    return res


def compute_border_indices(J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Parameters
    ----------
    J : int
        maximal subsampling by 2**J
    i0 : int
        start index of the original signal at the finest resolution
    i1 : int
        end index (excluded) of the original signal at the finest resolution

    Returns
    -------
    ind_start, ind_end: dictionaries with keys in [0, ..., J] such that the
        original signal is in padded_signal[ind_start[j]:ind_end[j]]
        after subsampling by 2**j
    """
    ind_start = {0: i0}
    ind_end = {0: i1}
    for j in range(1, J + 1):
        ind_start[j] = (ind_start[j - 1] // 2) + (ind_start[j - 1] % 2)
        ind_end[j] = (ind_end[j - 1] // 2) + (ind_end[j - 1] % 2)
    return ind_start, ind_end


def cast_psi(Psi, _type):
    """
    Casts the filters contained in Psi to the required type, by following
    the dictionary structure, and embed them into a variable as well.
    Warning: this function has border effects as it modifies the input pointer.

    Parameters
    ----------
    Psi : dictionary
        dictionary of dictionary of filters, should be psi1_fft or psi2_fft
    _type : torch type
        required type to cast the filters to. Should be a torch.FloatTensor

    Returns
    -------
    Nothing - function modifies the input
    """
    for key, item in Psi.items():
        for key2, item2 in Psi[key].items():
            if torch.is_tensor(item2):
                Psi[key][key2] = Variable(item2.type(_type).contiguous(),
                                          requires_grad=False)
            elif type(item2) == Variable:
                Psi[key][key2] = Variable(item2.data.type(_type).contiguous(),
                                          requires_grad=False)
            else:
                pass  # for the float entries


def cast_phi(Phi, _type):
    """
    Casts the filters contained in Phi to the required type, by following
    the dictionary structure, and embed them into a variable as well.
    Warning: this function has border effects as it modifies the input pointer.

    Parameters
    ----------
    Psi : dictionary
        dictionary of filters, should be phi_fft
    _type : torch type
        required type to cast the filters to. Should be a torch.FloatTensor

    Returns
    -------
    Nothing - function modifies the input
    """
    for key, item in Phi.items():
        if torch.is_tensor(item):
            Phi[key] = Variable(item.type(_type).contiguous(),
                                requires_grad=False)
        elif type(item) == Variable:
            Phi[key] = Variable(item.data.type(_type).contiguous(),
                                requires_grad=False)
        else:
            pass


def compute_padding(J_pad, T):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that 2**J_pad >= T

    Parameters
    ----------
    J_pad : int
        2**J_pad is the support of the padded signal
    T : int
        original signal support size

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)
    """
    T_pad = 2**J_pad
    if T_pad < T:
        raise ValueError('Padding support should be larger than the original' +
                         'signal size!')
    to_add = 2**J_pad - T
    pad_left = to_add // 2
    pad_right = to_add - pad_left
    if max(pad_left, pad_right) >= T:
        raise ValueError('Too large padding value, will lead to NaN errors')
    return pad_left, pad_right
