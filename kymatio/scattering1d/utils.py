import numpy as np
import math
from .filter_bank import (calibrate_scattering_filters, compute_temporal_support,
                          compute_minimum_required_length, gauss_1d, morlet_1d,
                          _recalibrate_psi_fr)

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

def compute_padding(J_pad, N):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that 2**J_pad >= N

    Parameters
    ----------
    J_pad : int
        2**J_pad is the support of the padded signal
    N : int
        original signal support size

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)
    """
    N_pad = 2**J_pad
    if N_pad < N:
        raise ValueError('Padding support should be larger than the original '
                         'signal size!')
    to_add = 2**J_pad - N
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
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
          - If `Q1==0`, will exclude `psi1_f` from computation.
          - If `Q2==0`, will exclude `psi2_f` from computation.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
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

    sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = \
        calibrate_scattering_filters(J_scattering, Q_temp, T, xi_min=xi_min)

    # compute psi1_f with greatest time support, if requested
    if Q1 >= 1:
        psi1_f_fn = lambda N: morlet_1d(N, xi1[-1], sigma1[-1],
                                        normalize=normalize, P_max=P_max, eps=eps)
    # compute psi2_f with greatest time support, if requested
    if Q2 >= 1:
        psi2_f_fn = lambda N: morlet_1d(N, xi2[-1], sigma2[-1],
                                        normalize=normalize, P_max=P_max, eps=eps)
    # compute lowpass
    phi_f_fn = lambda N: gauss_1d(N, sigma_low, P_max=P_max, eps=eps)

    # compute for all cases as psi's time support might exceed phi's
    kw = dict(criterion_amplitude=criterion_amplitude)
    N_min_phi = compute_minimum_required_length(phi_f_fn, N_init=N, **kw)
    phi_halfwidth = compute_temporal_support(
        phi_f_fn(N_min_phi).reshape(1, -1), **kw)

    if Q1 >= 1:
        N_min_psi1 = compute_minimum_required_length(psi1_f_fn, N_init=N, **kw)
        psi1_halfwidth = compute_temporal_support(
            psi1_f_fn(N_min_psi1).reshape(1, -1), **kw)
    else:
        psi1_halfwidth = -1  # placeholder
    if Q2 >= 1:
        N_min_psi2 = compute_minimum_required_length(psi2_f_fn, N_init=N, **kw)
        psi2_halfwidth = compute_temporal_support(
            psi2_f_fn(N_min_psi2).reshape(1, -1), **kw)
    else:
        psi2_halfwidth = -1

    # take maximum
    t_max = max(phi_halfwidth, psi1_halfwidth, psi2_halfwidth)

    # set min to pad based on maximum
    min_to_pad = int(1.2 * t_max)  # take a little extra to be safe
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
    Q : int >= 1 / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
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
    sigma_low, xi1, sigma1, j1, xi2, sigma2, j2 = \
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
    Q : int >= 1 / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
    J_pad : int
        2**J_pad == amount of temporal padding
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
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
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = \
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


def compute_meta_jtfs(J_pad, J, Q, J_fr, Q_fr, T, F, aligned, out_3D, out_type,
                      resample_filters_fr, average, sc_freq):
    """Get metadata on the Joint Time-Frequency Scattering transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.
    See below for more info.

    Parameters
    ----------
    J_pad : int
        2**J_pad == amount of temporal padding.
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : int >= 1 / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `2` or `1` for most applications.
    J_fr, Q_fr: int, int
        `J` and `Q` for frequential scattering.
    T : int
        Temporal support of temporal low-pass filter, controlling amount of
        imposed time-shift invariance and maximum subsampling
    F : int
        Temporal support of frequential low-pass filter, controlling amount of
        imposed frequency transposition invariance and subsampling
    out_type : str
         - `'dict:list'` or `'dict:array'`: meta is packed
           into respective pairs (e.g. `meta['n']['psi_t * phi_f'][1]`)
         - `'list'` or `'array'`: meta is flattened (e.g. `meta['n'][15]`).
    out_3D : bool
        - True: will reshape meta fields to match output structure:
          `(n_coeffs, n_freqs, meta_len)`.
        - False: pack flattened: `(n_coeffs * n_freqs, meta_len)`.
    resample_filters_fr : tuple[bool]
        See `help(TimeFrequencyScattering1D)`. Affects `xi`, `sigma`, and `j`.
    average : bool
        Only affects `S0`'s meta.
    sc_freq : `scattering1d.frontend.base_frontend._FrequencyScatteringBase`
        Frequential scattering object, storing pertinent attributes and filters.

    Returns
    -------
    meta : dictionary
        A dictionary with the following keys:

        - `'order`' : tensor
            A Tensor of length `C`, the total number of scattering
            coefficients, specifying the scattering order.
        - `'xi'` : tensor
            A Tensor of size `(C, 3)`, specifying the center
            frequency of the filter used at each order (padded with NaNs).
        - `'sigma'` : tensor
            A Tensor of size `(C, 3)`, specifying the frequency
            bandwidth of the filter used at each order (padded with NaNs).
        - `'j'` : tensor
            A Tensor of size `(C, 3)`, specifying the dyadic scale
            of the filter used at each order (padded with NaNs).
        - `'n'` : tensor
            A Tensor of size `(C, 3)`, specifying the indices of
            the filters used at each order (padded with NaNs).
            Lowpass filters in `phi_*` pairs are denoted via `-1`.
        - `'s'` : tensor
            A Tensor of length `C`, specifying the spin of
            each frequency scattering filter (+1=up, -1=down, 0=none).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.

    Computation and Structure
    -------------------------
    Computation replicates logic in `timefrequency_scattering()`. Meta values
    depend on:
        - out_3D (True only possible with average_fr=True)
        - average (only affects `S0`)
        - average_fr
        - average_fr_global
        - oversampling_fr
        - max_padding_fr
        - aligned (False meaningful (and tested) only with out_3D=True)
        - resample_psi_fr
        - resample_phi_fr
    and some of their interactions.
    """
    from .core.timefrequency_scattering import _get_stride

    def _get_compute_params(n2, n1_fr):
        """Reproduce exact logic in `timefrequency_scattering.py`."""
        # _frequency_scattering() or _frequency_lowpass() ####################
        # `n2 == -1` correctly indexes maximal amount of padding and unpadding
        pad_fr = (sc_freq.J_pad_fr_max if (aligned and out_3D) else
                  sc_freq.J_pad_fr[n2])
        shape_fr_padded = 2**pad_fr
        subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr

        if n1_fr != -1:
            j1_fr = sc_freq.psi1_f_fr_up[n1_fr]['j'][subsample_equiv_due_to_pad]
        else:
            j1_fr = sc_freq.phi_f_fr['j'][subsample_equiv_due_to_pad][0]

        total_conv_stride_over_U1 = _get_stride(j1_fr, pad_fr,
                                                subsample_equiv_due_to_pad,
                                                sc_freq, sc_freq.average_fr)

        if (n2 == -1 and n1_fr == -1):
            n1_fr_subsample = 0
        else:
            if sc_freq.average_fr:
                sub_adj = min(j1_fr, total_conv_stride_over_U1,
                              sc_freq.max_subsampling_before_phi_fr[n2])
            else:
                sub_adj = min(j1_fr, total_conv_stride_over_U1)
            n1_fr_subsample = max(sub_adj - sc_freq.oversampling_fr, 0)

        # _joint_lowpass() ####################################################
        # TODO private method in core?
        if sc_freq.average_fr or (n2 == -1 and n1_fr == -1):
            lowpass_subsample_fr = max(total_conv_stride_over_U1 -
                                       n1_fr_subsample -
                                       sc_freq.oversampling_fr, 0)
        else:
            lowpass_subsample_fr = 0

        total_subsample_so_far = subsample_equiv_due_to_pad + n1_fr_subsample
        total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr
        if out_3D:
            ind_start_fr = sc_freq.ind_start_fr_max[total_subsample_fr]
            ind_end_fr   = sc_freq.ind_end_fr_max[total_subsample_fr]
        else:
            ind_start_fr = sc_freq.ind_start_fr[n2][total_subsample_fr]
            ind_end_fr   = sc_freq.ind_end_fr[n2][total_subsample_fr]

        # TODO ref?
        total_conv_stride_over_U1 = n1_fr_subsample + lowpass_subsample_fr
        return (shape_fr_padded, total_conv_stride_over_U1,
                subsample_equiv_due_to_pad, ind_start_fr, ind_end_fr)

    def _get_fr_params(n1_fr, subsample_equiv_due_to_pad):
        k = subsample_equiv_due_to_pad
        if n1_fr != -1:
            if resample_psi_fr:
                p = xi1s_fr[n1_fr], sigma1s_fr[n1_fr], j1s_fr[n1_fr]
            else:
                p = (xi1s_fr_new[k][n1_fr], sigma1s_fr_new[k][n1_fr],
                     j1s_fr_new[k][n1_fr])
        else:
            if resample_phi_fr:
                p = 0, sigma_low, log2_F
            else:
                p = (xi1s_fr_phi[k], sigma1_fr_phi[k], j1s_fr_phi[k])
        xi1_fr, sigma1_fr, j1_fr = p
        return xi1_fr, sigma1_fr, j1_fr

    def _fill_n1_info(pair, n2, n1_fr, spin):
        # track S1 from padding to `_joint_lowpass()`
        (shape_fr_padded, total_conv_stride_over_U1, subsample_equiv_due_to_pad,
         ind_start_fr, ind_end_fr) = _get_compute_params(n2, n1_fr)

        # fetch xi, sigma for n2, n1_fr
        if n2 != -1:
            xi2, sigma2, j2 = xi2s[n2], sigma2s[n2], j2s[n2]
        else:
            xi2, sigma2, j2 = 0, sigma_low, log2_T
        xi1_fr, sigma1_fr, j1_fr = _get_fr_params(n1_fr,
                                                  subsample_equiv_due_to_pad)

        # distinguish between `key` and `n`
        n1_fr_n   = n1_fr if (n1_fr != -1) else inf
        n1_fr_key = n1_fr if (n1_fr != -1) else 0
        n2_n      = n2    if (n2    != -1) else inf
        n2_key    = n2    if (n2    != -1) else 0

        # global average pooling, all S1 collapsed into single point
        if sc_freq.average_fr_global:
            meta['order'][pair].append(2)
            meta['s'    ][pair].append((spin,))
            meta['j'    ][pair].append((j2,     j1_fr,     math.nan))
            meta['xi'   ][pair].append((xi2,    xi1_fr,    math.nan))
            meta['sigma'][pair].append((sigma2, sigma1_fr, math.nan))
            meta['n'    ][pair].append((n2_n,   n1_fr_n,   math.nan))
            meta['key'  ][pair].append((n2_key, n1_fr_key, 0))
            return

        fr_max = sc_freq.shape_fr[n2] if (n2 != -1) else len(xi1s)
        n1_step = 2 ** total_conv_stride_over_U1  # simulate subsampling
        for n1 in range(0, shape_fr_padded, n1_step):
            # simulate unpadding
            if n1 / n1_step < ind_start_fr:
                continue
            elif n1 / n1_step >= ind_end_fr:
                break

            if n1 >= fr_max:  # equivalently `j1 >= j2`
                # these are padded rows, no associated filters
                xi1, sigma1, j1 = math.nan, math.nan, math.nan
            else:
                xi1, sigma1, j1 = xi1s[n1], sigma1s[n1], j1s[n1]
            meta['order'][pair].append(2)
            meta['s'    ][pair].append((spin,))
            meta['j'    ][pair].append((j2,     j1_fr,     j1))
            meta['xi'   ][pair].append((xi2,    xi1_fr,    xi1))
            meta['sigma'][pair].append((sigma2, sigma1_fr, sigma1))
            meta['n'    ][pair].append((n2_n,   n1_fr_n,   n1))
            meta['key'  ][pair].append((n2_key, n1_fr_key, n1))

    # set params
    N, N_fr = 2**J_pad, 2**sc_freq.J_pad_fr_max_init
    xi_min = (2 / N)  # leftmost peak at bin 2
    xi_min_fr = (2 / 2**N_fr)
    log2_T = math.floor(math.log2(T))
    log2_F = math.floor(math.log2(F))
    # extract filter meta
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = \
        calibrate_scattering_filters(J, Q, T, xi_min=xi_min)
    sigma_low_fr, xi1s_fr, sigma1s_fr, j1s_fr, *_ = \
        calibrate_scattering_filters(J_fr, Q_fr, F, xi_min=xi_min_fr)

    # compute modified meta if `resample_=False`
    resample_psi_fr, resample_phi_fr = resample_filters_fr
    if not resample_psi_fr:
        xi1s_fr_new, sigma1s_fr_new, j1s_fr_new = _recalibrate_psi_fr(
            xi1s_fr, sigma1s_fr, j1s_fr, N_fr, sc_freq.alpha,
            sc_freq.subsampling_equiv_relative_to_max_padding)
    if not resample_phi_fr:
        phi_j1_frs = list(range(log2_F + 1))
        xi1s_fr_phi, sigma1_fr_phi, j1s_fr_phi = {}, {}, {}
        for j1_fr in phi_j1_frs:
            xi1s_fr_phi[j1_fr] = 0
            sigma1_fr_phi[j1_fr] = sigma_low * 2**j1_fr
            j1s_fr_phi[j1_fr] = log2_F - j1_fr

    meta = {}
    inf = -1  # placeholder for infinity
    nan = math.nan
    coef_names = (
        'S0',                  # (time)  zeroth order
        'S1',                  # (time)  first order
        'phi_t * phi_f',       # (joint) joint lowpass
        'phi_t * psi_f',       # (joint) time lowpass
        'psi_t * phi_f',       # (joint) freq lowpass
        'psi_t * psi_f_up',    # (joint) spin up
        'psi_t * psi_f_down',  # (joint) spin down
    )
    for field in ('order', 'xi', 'sigma', 'j', 'n', 's', 'key'):
        meta[field] = {name: [] for name in coef_names}

    # Zeroth-order
    meta['order']['S0'].append(0)
    meta['xi'   ]['S0'].append((nan, nan, 0         if average else nan))
    meta['sigma']['S0'].append((nan, nan, sigma_low if average else nan))
    meta['j'    ]['S0'].append((nan, nan, J         if average else nan))
    meta['n'    ]['S0'].append((nan, nan, inf       if average else nan))
    meta['s'    ]['S0'].append((nan,))
    meta['key'  ]['S0'].append((0, 0, 0))

    # First-order coeffs
    for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
        meta['order']['S1'].append(1)
        meta['xi'   ]['S1'].append((nan, nan, xi1))
        meta['sigma']['S1'].append((nan, nan, sigma1))
        meta['j'    ]['S1'].append((nan, nan, j1))
        meta['n'    ]['S1'].append((nan, nan, n1))
        meta['s'    ]['S1'].append((nan,))
        meta['key'  ]['S1'].append((0, 0, n1))

    S1_len = len(meta['n']['S1'])
    assert S1_len >= sc_freq.shape_fr_max

    # `phi_t * phi_f` coeffs
    _fill_n1_info('phi_t * phi_f', n2=-1, n1_fr=-1, spin=0)

    # `phi_t * psi_f` coeffs
    for n1_fr in range(len(j1s_fr)):
        _fill_n1_info('phi_t * psi_f', n2=-1, n1_fr=n1_fr, spin=0)

    # `psi_t * phi_f` coeffs
    for n2, j2 in enumerate(j2s):
        if j2 == 0:
            continue
        _fill_n1_info('psi_t * phi_f', n2, n1_fr=-1, spin=0)

    # `psi_t * psi_f` coeffs
    for spin in (1, -1):
        pair = ('psi_t * psi_f_up' if spin == 1 else
                'psi_t * psi_f_down')
        for n2, j2 in enumerate(j2s):
            if j2 == 0:
                continue
            for n1_fr, j1_fr in enumerate(j1s_fr):
                _fill_n1_info(pair, n2, n1_fr, spin=spin)

    array_fields = ['order', 'xi', 'sigma', 'j', 'n', 's']
    for field in array_fields:
        for pair, v in meta[field].items():
            meta[field][pair] = np.array(v)

    if out_3D:
      # reorder for 3D
      for field in array_fields:#meta:
        meta_len = 3 if field not in ('s', 'order') else 1
        for pair in meta[field]:
          if pair in ('S0', 'S1'):
              # simply expand dim for consistency, no 3D structure
              meta[field][pair] = meta[field][pair].reshape(-1, 1, meta_len)
              continue
          elif 'up' in pair or 'down' in pair:
              number_of_n2 = sum(j2 != 0 for j2 in j2s)
              number_of_n1_fr = len(j1s_fr)
          elif pair == 'psi_t * phi_f':
              number_of_n2 = sum(j2 != 0 for j2 in j2s)
              number_of_n1_fr = 1
          elif pair == 'phi_t * psi_f':
              number_of_n2 = 1
              number_of_n1_fr = len(j1s_fr)
          elif pair == 'phi_t * phi_f':
              number_of_n2 = 1
              number_of_n1_fr = 1
          n_coeffs = number_of_n2 * number_of_n1_fr
          meta[field][pair] = meta[field][pair].reshape(n_coeffs, -1, meta_len)

    if not out_type.startswith('dict'):
        # join pairs
        meta_flat = {f: np.concatenate([v for v in meta[f].values()], axis=0)
                     for f in meta if f not in ('key',)}
        meta_flat['key'] = meta['key']
        meta = meta_flat
    return meta
