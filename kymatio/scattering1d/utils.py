import numpy as np
import math
from .filter_bank import scattering_filter_factory, calibrate_scattering_filters

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

def compute_minimum_support_to_pad(T, J, Q, criterion_amplitude=1e-3,
                                       normalize='l1', r_psi=math.sqrt(0.5),
                                       sigma0=1e-1, alpha=5., P_max=5, eps=1e-7):


    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    T : int
        temporal size of the input signal
    J : int
        scale of the scattering
    Q : int
        number of wavelets per octave
    normalize : string, optional
        normalization type for the wavelets.
        Only `'l2'` or `'l1'` normalizations are supported.
        Defaults to `'l1'`
    criterion_amplitude: float `>0` and `<1`, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger criterion_amplitude, the smaller the padding size is.
        Defaults to `1e-3`
    r_psi : float, optional
        Should be `>0` and `<1`. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent
        wavelets).
        Defaults to `sqrt(0.5)`.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering,
        it is equal to :math:`\\frac{\\sigma_0}{2^J}`.
        Defaults to `1e-1`.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger the alpha, the more conservative the value of maximal
        subsampling is.
        Defaults to `5`.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic.
        `P_max = 5` is more than enough for double precision.
        Defaults to `5`.
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to `1e-7`.

    Returns
    -------
    min_to_pad: int
        minimal value to pad the signal on one size to avoid any
        boundary error.
    """
    J_tentative = int(np.ceil(np.log2(T)))
    _, _, _, t_max_phi = scattering_filter_factory(
        J_tentative, J, Q, normalize=normalize, to_torch=False,
        max_subsampling=0, criterion_amplitude=criterion_amplitude,
        r_psi=r_psi, sigma0=sigma0, alpha=alpha, P_max=P_max, eps=eps)
    min_to_pad = 3 * t_max_phi
    return min_to_pad


def precompute_size_scattering(J, Q, max_order=2, detail=False):
    """Get size of the scattering transform

    The number of scattering coefficients depends on the filter
    configuration and so can be calculated using a few of the scattering
    transform parameters.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : int >= 1
        The number of first-order wavelets per octave.
        Second-order wavelets are fixed to one wavelet per octave.
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.
    detail : boolean, optional
        Specifies whether to provide a detailed size (number of coefficient
        per order) or an aggregate size (total number of coefficients).

    Returns
    -------
    size : int or tuple
        If `detail` is `False`, returns the number of coefficients as an
        integer. If `True`, returns a tuple of size `max_order` containing
        the number of coefficients in each order.
    """
    sigma_low, xi1, sigma1, j1, xi2, sigma2, j2 = \
        calibrate_scattering_filters(J, Q)

    size_order0 = 1
    size_order1 = len(xi1)
    size_order2 = 0
    for n1 in range(len(xi1)):
        for n2 in range(len(xi2)):
            if j2[n2] > j1[n1]:
                size_order2 += 1
    if detail:
        if max_order == 2:
            return size_order0, size_order1, size_order2
        else:
            return size_order0, size_order1
    else:
        if max_order == 2:
            return size_order0 + size_order1 + size_order2
        else:
            return size_order0 + size_order1


def compute_meta_scattering(J, Q, max_order=2):
    """Get metadata on the transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : int >= 1
        The number of first-order wavelets per octave.
        Second-order wavelets are fixed to one wavelet per octave.
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.

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
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = \
        calibrate_scattering_filters(J, Q)

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
