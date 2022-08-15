import numpy as np
import math

def compute_border_indices(log2_T, J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Maximal subsampling is by `2**log2_T` if `T=None`, else by
    `2**max(log2_T, J)`. We compute indices up to latter to be sure.

    Parameters
    ----------
    log2_T : int
        Maximal subsampling by low-pass filtering is `2**log2_T`.
    J : int
        Maximal subsampling by band-pass filtering is `2**J`.
    i0 : int
        start index of the original signal at the finest resolution
    i1 : int
        end index (excluded) of the original signal at the finest resolution

    Returns
    -------
    ind_start, ind_end: dictionaries with keys in [0, ..., log2_T] such that the
        original signal is in padded_signal[ind_start[j]:ind_end[j]]
        after subsampling by 2**j
    """
    ind_start = {0: i0}
    ind_end = {0: i1}
    for j in range(1, max(log2_T, J) + 1):
        ind_start[j] = (ind_start[j - 1] // 2) + (ind_start[j - 1] % 2)
        ind_end[j] = (ind_end[j - 1] // 2) + (ind_end[j - 1] % 2)
    return ind_start, ind_end

def compute_padding(N, N_input):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that N >= N_input

    Parameters
    ----------
    N : int
        support of the padded signal
    N_input : int
        support of the unpadded signal

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)
    """
    if N < N_input:
        raise ValueError('Padding support should be larger than the original' +
                         'signal size!')
    to_add = N - N_input
    pad_left = to_add // 2
    pad_right = to_add - pad_left
    if max(pad_left, pad_right) >= N_input:
        raise ValueError('Too large padding value, will lead to NaN errors')
    return pad_left, pad_right


def parse_T(T, J, N_input, T_alias='T'):
    """
    Parses T in Scattering1D base frontend.
    Parses T and F in TimeFrequencyScattering base frontend.

    Parameters
    ----------
    T : None, string, integer 0, or float >= 1
        user-provided T value
    J : int
        user-provided J value
    N_input : int
        input size
    T_alias : string
        Used for printing error messages.
        Typically 'T' (default) or 'F' (in TimeFrequencyScattering).

    Returns
    -------
    T_parsed : int
        (2**J) if T is None, zero, or 'global'; user-provided T otherwise
    average : string
        'global' if T is 'global'; False if T is zero; 'local' otherwise
    """
    if T is None:
        return 2 ** J, 'local'
    elif T == 'global':
        return 2 ** J, 'global'
    elif T > N_input:
        raise ValueError("The support {} of the low-pass filter cannot exceed "
            "input length (got {} > {}). For large averaging size, consider "
            "passing {}='global'.".format(T_alias, T, N_input, T_alias))
    elif T == 0:
        return 2 ** J, False
    elif T < 1:
        raise ValueError("{} must be ==0 or >=1 (got {})".format(T_alias, T))
    else:
        return T, 'local'
