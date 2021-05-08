import numpy as np
import math
from .filter_bank import (calibrate_scattering_filters, compute_temporal_support,
                          compute_minimum_required_length, gauss_1d, morlet_1d)

def compute_border_indices(log2_T, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Parameters
    ----------
    log2_T : int
        maximal subsampling by 2**log2_T
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
    for j in range(1, log2_T + 1):
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
        raise ValueError('Padding support should be larger than the original '
                         'signal size!')
    to_add = 2**J_pad - T
    pad_left = to_add // 2
    pad_right = to_add - pad_left
    return pad_left, pad_right

def compute_minimum_support_to_pad(N, J, Q, T, criterion_amplitude=1e-3,
                                       normalize='l1', r_psi=math.sqrt(0.5),
                                       sigma0=1e-1, alpha=5., P_max=5, eps=1e-7,
                                       pad_mode='reflect'):


    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    N : int
        temporal size of the input signal
    J : int
        scale of the scattering
    Q : int >= 1
        The number of first-order wavelets per octave.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and subsampling
    Q2 : int >= 0  # TODO
        The number of second-order wavelets per octave.
        If 0, will exclude `psi2` from computation.
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
    pad_mode : str
        Name of padding used. If 'zero', will halve `min_to_pad`, else no effect.

    Returns
    -------
    min_to_pad: int
        minimal value to pad the signal on one size to avoid any
        boundary error.
    """
    # compute params for calibrating, & calibrate
    J_tentative = int(np.ceil(np.log2(N)))
    J_support = J_tentative
    J_scattering = J

    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    Q_temp = (max(Q1, 1), max(Q2, 1))  # don't pass in zero
    N = 2 ** J_support
    xi_min = (2 / N)  # leftmost peak at bin 2

    sigma_low, xi1, sigma1, j1s, is_cqt1, xi2, sigma2, j2s, is_cqt2 = \
        calibrate_scattering_filters(J_scattering, Q_temp, T, xi_min=xi_min)

    def last_cqt(is_cqt):
        """Begin iterating from last CQT wavelet."""
        return (is_cqt.index(False) if False in is_cqt else len(is_cqt)) - 1

    # compute psi1_f with greatest time support, if requested
    if Q1 >= 1:
        # TODO subsampled variant
        n1_last_cqt = last_cqt(is_cqt1)
        for n1 in range(n1_last_cqt, len(j1s)):
            try:
                psi1_f_fn = lambda N: morlet_1d(
                    N, xi1[n1], sigma1[n1], normalize=normalize, P_max=P_max,
                    eps=eps)
                _ = psi1_f_fn(N)
            except ValueError as e:
                if is_cqt1[n1]:
                    raise e
                break
        psi1_f_fn_widest = psi1_f_fn

    # compute psi2_f with greatest time support, if requested
    if Q2 >= 1:
        n2_last_cqt = last_cqt(is_cqt2)
        for n2 in range(n2_last_cqt, len(j2s)):
            try:
                psi2_f_fn = lambda N: morlet_1d(
                    N, xi2[n2], sigma2[n2], normalize=normalize, P_max=P_max,
                    eps=eps)
                _ = psi2_f_fn(N)
            except ValueError as e:
                if is_cqt2[n2]:
                    raise e
                break
        psi2_f_fn_widest = psi2_f_fn

    # compute lowpass
    phi_f_fn = lambda N: gauss_1d(N, sigma_low, P_max=P_max, eps=eps)

    # compute for all cases as psi's time support might exceed phi's
    kw = dict(N_init=N, criterion_amplitude=criterion_amplitude)
    N_min_phi = compute_minimum_required_length(phi_f_fn, **kw)
    if Q1 >= 1:
        N_min_psi1 = compute_minimum_required_length(psi1_f_fn, **kw)
    else:
        N_min_psi1 = -2  # placeholder
    if Q2 >= 1:
        N_min_psi2 = compute_minimum_required_length(psi2_f_fn, **kw)
    else:
        N_min_psi2 = -2

    # take maximum
    if N_min_phi == max(N_min_phi, N_min_psi1, N_min_psi2):
        p_fr = phi_f_fn(N_min_phi)
    elif N_min_psi1 == max(N_min_phi, N_min_psi1, N_min_psi2):
        p_fr = psi1_f_fn_widest(N_min_psi1)
    else:
        p_fr = psi2_f_fn_widest(N_min_psi2)
    t_max = compute_temporal_support(p_fr.reshape(1, -1),
                                     criterion_amplitude=criterion_amplitude)

    # set min to pad based on maximum
    min_to_pad = int(1.2 * t_max)
    if pad_mode == 'zero':
        min_to_pad //= 2

    # return results
    return min_to_pad


def precompute_size_scattering(J, Q, T, max_order=2, detail=False):
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
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and subsampling
    Q2 : int >= 1
        The number of second-order wavelets per octave.
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
    sigma_low, xi1, sigma1, j1, is_cqt1, xi2, sigma2, j2, is_cqt2 = \
        calibrate_scattering_filters(J, Q, T)

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


def compute_meta_scattering(J, Q, J_pad, T, max_order=2):
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
    J_pad : int
        2**J_pad == amount of temporal padding
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and subsampling
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.
    xi_min : float, optional
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

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
    xi_min = (2 / 2**J_pad)  # leftmost peak at bin 2
    # TODO meta won't match output if non-CQT are dropped
    sigma_low, xi1s, sigma1s, j1s, _, xi2s, sigma2s, j2s, _ = \
        calibrate_scattering_filters(J, Q, T, xi_min=xi_min)

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


def compute_meta_jtfs(J, Q, J_pad, J_pad_fr_max, T, F, J_fr, Q_fr):
    """Get metadata on the Joint Time-Frequency Scattering transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : int >= 1
        The number of first-order wavelets per octave.
    J_pad : int
        2**J_pad == amount of temporal padding
    J_pad_fr_max: int
        2**J_pad_fr_max == maximum amount of frequential padding
    T : int
        temporal support of temporal low-pass filter, controlling amount of
        imposed time-shift invariance and subsampling
    F : int
        temporal support of frequential low-pass filter, controlling amount of
        imposed frequency transposition invariance and subsampling
    Q2 : int >= 1
        The number of second-order wavelets per octave.
    J_fr, Q_fr: int, int
        `J` and `Q` for frequential scattering.

    Returns
    -------
    meta : dictionary
        A dictionary with the following keys:

        - `'order`' : tensor
            A Tensor of length `C`, the total number of scattering
            coefficients, specifying the scattering order.
        - `'xi'` : tensor
            A Tensor of size `(C, 2)`, specifying the center
            frequency of the filter used at each order (padded with NaNs).
        - `'sigma'` : tensor
            A Tensor of size `(C, 2)`, specifying the frequency
            bandwidth of the filter used at each order (padded with NaNs).
        - `'j'` : tensor
            A Tensor of size `(C, 2)`, specifying the dyadic scale
            of the filter used at each order (padded with NaNs).
        - `'n'` : tensor
            A Tensor of size `(C, 2)`, specifying the indices of
            the filters used at each order (padded with NaNs).
        - `'s'` : tensor
            A Tensor of length `C`, specifying the spin of
            each frequency scattering filter (+1=up, -1=down, 0=none).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.
    """
    xi_min = (2 / 2**J_pad)  # leftmost peak at bin 2
    xi_min_fr = (2 / 2**J_pad_fr_max)
    sigma_low, xi1s, sigma1s, j1s, _, xi2s, sigma2s, j2s, _ = \
        calibrate_scattering_filters(J, Q, T, xi_min=xi_min)
    sigma_low_fr, xi1s_fr, sigma1s_fr, j1s_fr, *_ = \
        calibrate_scattering_filters(J_fr, Q_fr, F, xi_min=xi_min_fr)

    meta = {}
    inf = -1  # placeholder for infinity

    coef_names = ('S0',                  # (time)  zeroth order
                  'S1',                  # (time)  first order
                  'psi_t * psi_f_up',    # (joint) spin up
                  'psi_t * psi_f_down',  # (joint) spin down
                  'psi_t * phi_f',       # (joint) freq lowpass
                  'phi_t * psi_f',       # (joint) time lowpass
                  'phi_t * phi_f')       # (joint) joint lowpass
    for field in ('order', 'xi', 'sigma', 'j', 'n', 's', 'key'):
        meta[field] = {name: [] for name in coef_names}

    # Zeroth-order
    meta['order']['S0'].append(0)
    for field in meta:
        if field != 'order':
            meta[field]['S0'].append(())

    # First-order coeffs
    for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
        meta['order']['S1'].append(1)
        meta['xi'   ]['S1'].append((xi1,))
        meta['sigma']['S1'].append((sigma1,))
        meta['j'    ]['S1'].append((j1,))
        meta['n'    ]['S1'].append((n1,))
        meta['s'    ]['S1'].append(())
        meta['key'  ]['S1'].append((n1,))

    # TODO drop `order`?
    # TODO -1 or inf doesn't make sense for `key`
    # TODO drop `key`? no "non-vectorized" output, and it doesn't do as stated
    # TODO meta won't match output if non-CQT are dropped
    # Frequential lowpass over first-order
    meta['order']['phi_t * phi_f'].append(0)
    for field in meta:
        if field != 'order':
            meta[field]['phi_t * phi_f'].append(())

    # `psi_t * psi_f` coeffs
    for spin in (1, -1):
        k = ('psi_t * psi_f_up' if spin == 1 else
             'psi_t * psi_f_down')
        for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
            if j2 == 0:
                continue
            for (n1_fr, (xi1_fr, sigma1_fr, j1_fr)
                 ) in enumerate(zip(xi1s_fr, sigma1s_fr, j1s_fr)):
                meta['order'][k].append(1)
                meta['xi'   ][k].append((xi2, xi1_fr,))
                meta['sigma'][k].append((sigma2, sigma1_fr,))
                meta['j'    ][k].append((j2, j1_fr))
                meta['n'    ][k].append((n2, n1_fr))
                meta['s'    ][k].append((spin,))
                meta['key'  ][k].append((n2, n1_fr))

    # `psi_t * phi_f` coeffs
    log2_F = math.floor(math.log2(F))
    for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
        if j2 == 0:
            continue
        meta['order']['psi_t * phi_f'].append(1)
        meta['xi'   ]['psi_t * phi_f'].append((xi2, 0))
        meta['sigma']['psi_t * phi_f'].append((sigma2, sigma_low_fr))
        meta['j'    ]['psi_t * phi_f'].append((j2, log2_F))
        meta['n'    ]['psi_t * phi_f'].append((n2, inf))
        meta['s'    ]['psi_t * phi_f'].append((0,))
        meta['key'  ]['psi_t * phi_f'].append((n2, inf))

    # `phi_t * psi_f` coeffs
    log2_T = math.floor(math.log2(T))
    for (n1_fr, (xi1_fr, sigma1_fr, j1_fr)
         ) in enumerate(zip(xi1s_fr, sigma1s_fr, j1s_fr)):
        meta['order']['phi_t * psi_f'].append(1)
        meta['xi'   ]['phi_t * psi_f'].append((0, xi1_fr,))
        meta['sigma']['phi_t * psi_f'].append((sigma_low, sigma1_fr,))
        meta['j'    ]['phi_t * psi_f'].append((log2_T, j1_fr))
        meta['n'    ]['phi_t * psi_f'].append((inf, n1_fr))
        meta['s'    ]['phi_t * psi_f'].append((0,))
        meta['key'  ]['phi_t * psi_f'].append((inf, n1_fr))

    pad_fields = ['xi', 'sigma', 'j', 'n']
    pad_len = 2
    for field in pad_fields:
        for name, v in meta[field].items():
            meta[field][name] = [x + (math.nan,) * (pad_len - len(x)) for x in v]
    # spin is of pad_len=1
    for name, v in meta['s'].items():
        meta['s'][name] = [(math.nan,) if s == () else s for s in v]

    array_fields = ['order', 'xi', 'sigma', 'j', 'n', 's']
    for field in array_fields:
        for name, v in meta[field].items():
            meta[field][name] = np.array(v)

    return meta
