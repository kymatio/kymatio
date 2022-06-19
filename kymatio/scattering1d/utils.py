import numpy as np
import math
from .filter_bank import scattering_filter_factory, compute_params_filterbank

def compute_border_indices(log2_T, J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Maximal subsampling is by `2**log2_T` if `average=True`, else by
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


def precompute_size_scattering(J, Q, T, max_order, r_psi, sigma0, alpha):
    """Get size of the scattering transform

    The number of scattering coefficients depends on the filter
    configuration and so can be calculated using a few of the scattering
    transform parameters.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : tuple
        number of wavelets per octave at the first and second order 
        Q = (Q1, Q2). Q1 and Q2 are both int >= 1.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    max_order : int
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
    sigma0 : float
        parameter controlling the frequential width of the low-pass filter at
        j=0; at a an absolute J, it is equal to sigma0 / 2**J.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is.

    Returns
    -------
    size : tuple
        A tuple of size `1+max_order` containing the number of coefficients in
        orders zero up to `max_order`, both included.
    """
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s, j1s = compute_params_filterbank(sigma_min, Q1, alpha, r_psi)
    xi2s, sigma2s, j2s = compute_params_filterbank(sigma_min, Q2, alpha, r_psi)

    sizes = [1, len(xi1s)]
    size_order2 = 0
    for n1 in range(len(xi1s)):
        for n2 in range(len(xi2s)):
            if j2s[n2] > j1s[n1]:
                size_order2 += 1

    if max_order == 2:
        sizes.append(size_order2)
    return sizes


def compute_meta_scattering(J, Q, T, max_order, r_psi, sigma0, alpha):
    """Get metadata on the transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : tuple
        number of wavelets per octave at the first and second order 
        Q = (Q1, Q2). Q1 and Q2 are both int >= 1.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    max_order : int
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
    sigma0 : float
        parameter controlling the frequential width of the low-pass filter at
        j=0; at a an absolute J, it is equal to sigma0 / 2**J.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is.

    Returns
    -------
    meta : dictionary
        A dictionary with the following keys:

        - `'order`' : tensor
            A Tensor of length `C`, the total number of scattering
            coefficients, specifying the scattering order.
        - `'xi'` : tensor
            A Tensor of size `(C, max_order)`, specifying the center
            frequency of the filter used at each order (padded with NaNs).
        - `'sigma'` : tensor
            A Tensor of size `(C, max_order)`, specifying the frequency
            bandwidth of the filter used at each order (padded with NaNs).
        - `'j'` : tensor
            A Tensor of size `(C, max_order)`, specifying the dyadic scale
            of the filter used at each order (padded with NaNs).
        - `'n'` : tensor
            A Tensor of size `(C, max_order)`, specifying the indices of
            the filters used at each order (padded with NaNs).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.
    """
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s, j1s = compute_params_filterbank(sigma_min, Q1, alpha, r_psi)
    xi2s, sigma2s, j2s = compute_params_filterbank(sigma_min, Q2, alpha, r_psi)

    meta = {}

    meta['order'] = [[], [], []]
    meta['xi'] = [[], [], []]
    meta['sigma'] = [[], [], []]
    meta['j'] = [[], [], []]
    meta['n'] = [[], [], []]
    meta['key'] = [[], [], []]

    meta['order'][0].append(0)
    meta['xi'][0].append(())
    meta['sigma'][0].append(())
    meta['j'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())

    for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))

        if max_order < 2:
            continue

        for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
            if j2 > j1:
                meta['order'][2].append(2)
                meta['xi'][2].append((xi1, xi2))
                meta['sigma'][2].append((sigma1, sigma2))
                meta['j'][2].append((j1, j2))
                meta['n'][2].append((n1, n2))
                meta['key'][2].append((n1, n2))

    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]

    pad_fields = ['xi', 'sigma', 'j', 'n']
    pad_len = max_order

    for field in pad_fields:
        meta[field] = [x + (math.nan,) * (pad_len - len(x)) for x in meta[field]]

    array_fields = ['order', 'xi', 'sigma', 'j', 'n']

    for field in array_fields:
        meta[field] = np.array(meta[field])

    return meta
