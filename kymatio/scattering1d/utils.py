import numpy as np
import torch



def ceiling_strict(s):
    """
    String ceiling of an input.
    """
    c = np.ceil(s)
    if c == s:
        return int(s + 1)
    else:
        return int(c)



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
    the dictionary structure.

    Parameters
    ----------
    Psi : dictionary
        dictionary of dictionary of filters, should be psi1_f or psi2_f
    _type : torch type
        required type to cast the filters to. Should be a torch.FloatTensor

    Returns
    -------
    Nothing - function modifies the input
    """
    for filt in Psi:
        for k in filt.keys():
            if torch.is_tensor(filt[k]):
                filt[k] = filt[k].type(_type).contiguous().requires_grad_(False)
            else:
                pass  # for the float entries


def cast_phi(Phi, _type):
    """
    Casts the filters contained in Phi to the required type, by following
    the dictionary structure.

    Parameters
    ----------
    Psi : dictionary
        dictionary of filters, should be phi_f
    _type : torch type
        required type to cast the filters to. Should be a torch.FloatTensor

    Returns
    -------
    Nothing - function modifies the input
    """
    for k in Phi.keys():
        if torch.is_tensor(Phi[k]):
            Phi[k] = Phi[k].type(_type).contiguous().requires_grad_(False)
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

