import numpy as np
import math
from .filter_bank import (calibrate_scattering_filters, compute_temporal_support,
                          compute_minimum_required_length, gauss_1d, morlet_1d,
                          _recalibrate_psi_fr)

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
    pad_right = to_add // 2
    pad_left = to_add - pad_right
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
    # compute without limit, will find necessary pad_psi then check in factories
    xi_min = -1

    sigma_low, xi1, sigma1, j1s, _, xi2, sigma2, j2s, _ = \
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
    ca = dict(criterion_amplitude=criterion_amplitude)
    N_min_phi = compute_minimum_required_length(phi_f_fn, N_init=N, **ca)
    phi_halfwidth = compute_temporal_support(phi_f_fn(N_min_phi)[None], **ca)

    if Q1 >= 1:
        N_min_psi1 = compute_minimum_required_length(psi1_f_fn, N_init=N, **ca)
        psi1_halfwidth = compute_temporal_support(psi1_f_fn(N_min_psi1)[None],
                                                  **ca)
    else:
        psi1_halfwidth = -1  # placeholder
    if Q2 >= 1:
        N_min_psi2 = compute_minimum_required_length(psi2_f_fn, N_init=N, **ca)
        psi2_halfwidth = compute_temporal_support(psi2_f_fn(N_min_psi2)[None],
                                                  **ca)
    else:
        psi2_halfwidth = -1

    # set min to pad based on each
    pads = (phi_halfwidth, psi1_halfwidth, psi2_halfwidth)

    # can pad half as much
    if pad_mode == 'zero':
        pads = [p//2 for p in pads]
    pad_phi, pad_psi1, pad_psi2 = pads
    # set main quantity as the max of all
    min_to_pad = max(pads)

    # return results
    return min_to_pad, pad_phi, pad_psi1, pad_psi2


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
    sigma_low, xi1, sigma1, j1, _, xi2, sigma2, j2, _ = \
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


def compute_meta_scattering(J, Q, J_pad, T, r_psi=math.sqrt(.5), max_order=2):
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
    r_psi : float
        Filter redundancy.
        See `help(kymatio.scattering1d.filter_bank.calibrate_scattering_filters)`.
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
        - `'is_cqt'` : tensor
            A tensor of size `(C, max_order)`, specifying whether the filter
            was constructed per Constant Q Transform (padded with NaNs).
        - `'n'` : tensor
            A Tensor of size `(C, max_order)`, specifying the indices of
            the filters used at each order (padded with NaNs).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.
    """
    xi_min = (2 / 2**J_pad)  # leftmost peak at bin 2
    sigma_low, xi1s, sigma1s, j1s, is_cqt1s, xi2s, sigma2s, j2s, is_cqt2s = \
        calibrate_scattering_filters(J, Q, T, r_psi=r_psi, xi_min=xi_min)
    log2_T = math.floor(math.log2(T))

    meta = {}

    meta['order'] = [[], [], []]
    meta['xi'] = [[], [], []]
    meta['sigma'] = [[], [], []]
    meta['j'] = [[], [], []]
    meta['is_cqt'] = [[], [], []]
    meta['n'] = [[], [], []]
    meta['key'] = [[], [], []]

    meta['order'][0].append(0)
    meta['xi'][0].append((0,))
    meta['sigma'][0].append((sigma_low,))
    meta['j'][0].append((log2_T,))
    meta['is_cqt'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())

    for (n1, (xi1, sigma1, j1, is_cqt1)
         ) in enumerate(zip(xi1s, sigma1s, j1s, is_cqt1s)):
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['is_cqt'][1].append((is_cqt1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))

        if max_order < 2:
            continue

        for (n2, (xi2, sigma2, j2, is_cqt2)
             ) in enumerate(zip(xi2s, sigma2s, j2s, is_cqt2s)):
            if j2 > j1:
                meta['order'][2].append(2)
                meta['xi'][2].append((xi1, xi2))
                meta['sigma'][2].append((sigma1, sigma2))
                meta['j'][2].append((j1, j2))
                meta['is_cqt'][2].append((is_cqt1, is_cqt2))
                meta['n'][2].append((n1, n2))
                meta['key'][2].append((n1, n2))

    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]

    pad_fields = ['xi', 'sigma', 'j', 'is_cqt', 'n']
    pad_len = max_order

    for field in pad_fields:
        meta[field] = [x + (math.nan,) * (pad_len - len(x)) for x in meta[field]]

    array_fields = ['order', 'xi', 'sigma', 'j', 'is_cqt', 'n']

    for field in array_fields:
        meta[field] = np.array(meta[field])

    return meta


def compute_meta_jtfs(J_pad, J, Q, J_fr, Q_fr, T, F, aligned, out_3D, out_type,
                      out_exclude, sampling_filters_fr, average, average_global,
                      average_global_phi, oversampling, r_psi, scf):
    """Get metadata on the Joint Time-Frequency Scattering transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.
    See below for more info.

    Parameters
    ----------
    J_pad : int
        2**J_pad == amount of temporal padding.

    J, Q, J_fr, T, F: int, int, int, int, int
        See `help(kymatio.scattering1d.TimeFrequencyScattering1D)`.
        Control physical meta of bandpass and lowpass filters (xi, sigma, etc).

    out_3D : bool
        - True: will reshape meta fields to match output structure:
          `(n_coeffs, n_freqs, meta_len)`.
        - False: pack flattened: `(n_coeffs * n_freqs, meta_len)`.

    out_type : str
         - `'dict:list'` or `'dict:array'`: meta is packed
           into respective pairs (e.g. `meta['n']['psi_t * phi_f'][1]`)
         - `'list'` or `'array'`: meta is flattened (e.g. `meta['n'][15]`).

    out_exclude : list/tuple[str]
        Names of coefficient pairs to exclude from meta.

    sampling_filters_fr : tuple[str]
        See `help(TimeFrequencyScattering1D)`. Affects `xi`, `sigma`, and `j`.

    average : bool
        Affects `S0`'s meta, and temporal stride meta.

    average_global : bool
        Affects `S0`'s meta, and temporal stride meta.

    average_global_phi : bool
        Affects joint temporal stride meta.

    oversampling : int
        Affects temporal stride meta.

    scf : `scattering1d.frontend.base_frontend._FrequencyScatteringBase`
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
            of the filter used at each order (padded with NaNs), excluding
            lowpass filtering (unless it was the only filtering).
        - `'is_cqt'` : tensor
            A tensor of size `(C, max_order)`, specifying whether the filter
            was constructed per Constant Q Transform (padded with NaNs).
        - `'n'` : tensor
            A Tensor of size `(C, 3)`, specifying the indices of
            the filters used at each order (padded with NaNs).
            Lowpass filters in `phi_*` pairs are denoted via `-1`.
        - `'s'` : tensor
            A Tensor of length `C`, specifying the spin of
            each frequency scattering filter (+1=up, -1=down, 0=none).
        - `'stride'` : tensor
            A Tensor of size `(C, 2)`, specifying the total temporal and
            frequential convolutional stride (i.e. subsampling) of resulting
            coefficient (including lowpass filtering).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.

        In case of `out_3D=True`, for joint pairs, will reshape each field into
        `(n_coeffs, C, meta_len)`, where `n_coeffs` is the number of joint slices
        in the pair, and `meta_len` is the existing `shape[-1]` (1, 2, or 3).

    Computation and Structure
    -------------------------
    Computation replicates logic in `timefrequency_scattering1d()`. Meta values
    depend on:
        - out_3D (True only possible with `average and average_fr`)
        - average
        - average_global
        - average_global_phi
        - average_fr
        - average_fr_global
        - average_fr_global_phi
        - oversampling
        - oversampling_fr
        - max_padding_fr
        - aligned
        - sampling_psi_fr
        - sampling_phi_fr
    and some of their interactions.
    """
    from .core.timefrequency_scattering1d import _get_stride

    def _get_compute_params(n2, n1_fr):
        """Reproduce exact logic in `timefrequency_scattering1d.py`."""
        # _frequency_scattering() or _frequency_lowpass() ####################
        # `n2 == -1` correctly indexes maximal amount of padding and unpadding
        pad_fr = (scf.J_pad_frs_max if (aligned and out_3D) else
                  scf.J_pad_frs[n2])
        N_fr_padded = 2**pad_fr
        subsample_equiv_due_to_pad = scf.J_pad_frs_max_init - pad_fr

        if n1_fr != -1:
            j1_fr = scf.psi1_f_fr_up[n1_fr]['j'][subsample_equiv_due_to_pad]
        else:
            j1_fr = (scf.phi_f_fr['j'][subsample_equiv_due_to_pad]
                     if not scf.average_fr_global_phi else
                     min(pad_fr, scf.log2_F))

        # `* phi_f` pairs always behave per `average_fr=True` (if `not aligned`)
        if not scf.aligned:
            _average_fr = (True if n1_fr == -1 else scf.average_fr)
        else:
            _average_fr = scf.average_fr
        total_conv_stride_over_U1 = _get_stride(j1_fr, pad_fr,
                                                subsample_equiv_due_to_pad,
                                                scf, _average_fr)
        if n2 == -1 and n1_fr == -1:
            n1_fr_subsample = 0
        elif n1_fr == -1 and scf.average_fr_global_phi:
            n1_fr_subsample = total_conv_stride_over_U1
        else:
            if scf.average_fr:
                sub_adj = min(j1_fr, total_conv_stride_over_U1,
                              scf.max_subsample_before_phi_fr[n2])
            else:
                sub_adj = min(j1_fr, total_conv_stride_over_U1)
            n1_fr_subsample = max(sub_adj - scf.oversampling_fr, 0)

        # _joint_lowpass() ####################################################
        global_averaged_fr = (scf.average_fr_global if n1_fr != -1 else
                              scf.average_fr_global_phi)
        if global_averaged_fr:
            lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample
        elif scf.average_fr or (n2 == -1 and n1_fr == -1):
            lowpass_subsample_fr = max(total_conv_stride_over_U1 -
                                       n1_fr_subsample -
                                       scf.oversampling_fr, 0)
        else:
            lowpass_subsample_fr = 0

        total_conv_stride_over_U1_realized = (n1_fr_subsample +
                                              lowpass_subsample_fr)
        # unpad params, used only if `not global_averaged_fr`
        # (except for energy correction, which isn't done here)
        if out_3D:
            pad_ref = (scf.J_pad_frs_min if aligned else
                       scf.J_pad_frs_max)
            subsample_equiv_due_to_pad_ref = (scf.J_pad_frs_max_init -
                                              pad_ref)
            # ensure stride is zero if `not average and aligned`
            average_fr = bool(scf.average_fr or not aligned)
            stride_ref = _get_stride(
                None, pad_ref, subsample_equiv_due_to_pad_ref, scf, average_fr)
            stride_ref = max(stride_ref - scf.oversampling_fr, 0)
            ind_start_fr = scf.ind_start_fr_max[stride_ref]
            ind_end_fr   = scf.ind_end_fr_max[  stride_ref]
        else:
            _stride = total_conv_stride_over_U1_realized
            ind_start_fr = scf.ind_start_fr[n2][_stride]
            ind_end_fr   = scf.ind_end_fr[  n2][_stride]

        return (N_fr_padded, total_conv_stride_over_U1_realized,
                n1_fr_subsample, subsample_equiv_due_to_pad,
                ind_start_fr, ind_end_fr, global_averaged_fr)

    def _get_fr_params(n1_fr, subsample_equiv_due_to_pad):
        k = subsample_equiv_due_to_pad
        if n1_fr != -1:
            if sampling_psi_fr in ('resample', 'exclude'):
                p = (xi1s_fr[n1_fr], sigma1s_fr[n1_fr], j1s_fr[n1_fr],
                     is_cqt1_frs[n1_fr])
            elif sampling_psi_fr == 'recalibrate':
                p = (xi1s_fr_new[k][n1_fr], sigma1s_fr_new[k][n1_fr],
                     j1s_fr_new[k][n1_fr], is_cqt1_frs_new[k][n1_fr])
        else:
            if not scf.average_fr_global_phi:
                p = [m[k] for m in (xi1s_fr_phi, sigma1_fr_phi, j1s_fr_phi)
                     ] + [nan]
            else:
                pad_fr = scf.J_pad_frs_max_init - subsample_equiv_due_to_pad
                j1_fr = min(pad_fr, scf.log2_F)
                p = (0, scf.sigma0 / 2**j1_fr, j1_fr, nan)

        xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr = p
        return xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr

    def _exclude_excess_scale(n2, n1_fr):
        if scf.sampling_psi_fr != 'exclude' or n1_fr == -1:
            return False

        pad_fr = (scf.J_pad_frs_max if (aligned and out_3D) else
                  scf.J_pad_frs[n2])
        subsample_equiv_due_to_pad = scf.J_pad_frs_max_init - pad_fr
        j0s = [k for k in scf.psi1_f_fr_up[n1_fr] if isinstance(k, int)]
        if subsample_equiv_due_to_pad not in j0s:
            return True

        width = scf.psi1_f_fr_up[n1_fr]['width'][subsample_equiv_due_to_pad]
        if width > scf.N_frs[n2]:
            return True
        return False

    def _fill_n1_info(pair, n2, n1_fr, spin):
        if _exclude_excess_scale(n2, n1_fr):
            return

        # track S1 from padding to `_joint_lowpass()`
        (N_fr_padded, total_conv_stride_over_U1_realized, n1_fr_subsample,
         subsample_equiv_due_to_pad, ind_start_fr, ind_end_fr, global_averaged_fr
         ) = _get_compute_params(n2, n1_fr)

        # fetch xi, sigma for n2, n1_fr
        if n2 != -1:
            xi2, sigma2, j2, is_cqt2 = (xi2s[n2], sigma2s[n2], j2s[n2],
                                        is_cqt2s[n2])
        else:
            xi2, sigma2, j2, is_cqt2 = 0, sigma_low, log2_T, nan
        xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr = _get_fr_params(
            n1_fr, subsample_equiv_due_to_pad)

        # get temporal stride info
        global_averaged = (average_global if n2 != -1 else
                           average_global_phi)
        if global_averaged:
            total_conv_stride_tm = log2_T
        else:
            k1_plus_k2 = max(min(j2, log2_T) - oversampling, 0)
            if average:
                k2_tm_J = max(log2_T - k1_plus_k2 - oversampling, 0)
                total_conv_stride_tm = k1_plus_k2 + k2_tm_J
            else:
                total_conv_stride_tm = k1_plus_k2
        stride = (total_conv_stride_over_U1_realized, total_conv_stride_tm)

        # distinguish between `key` and `n`
        n1_fr_n   = n1_fr if (n1_fr != -1) else inf
        n1_fr_key = n1_fr if (n1_fr != -1) else 0
        n2_n      = n2    if (n2    != -1) else inf
        n2_key    = n2    if (n2    != -1) else 0

        # global average pooling, all S1 collapsed into single point
        if global_averaged_fr:
            meta['order' ][pair].append(2)
            meta['xi'    ][pair].append((xi2,     xi1_fr,     nan))
            meta['sigma' ][pair].append((sigma2,  sigma1_fr,  nan))
            meta['j'     ][pair].append((j2,      j1_fr,      nan))
            meta['is_cqt'][pair].append((is_cqt2, is_cqt1_fr, nan))
            meta['n'     ][pair].append((n2_n,    n1_fr_n,    nan))
            meta['s'     ][pair].append((spin,))
            meta['stride'][pair].append(stride)
            meta['key'   ][pair].append((n2_key,  n1_fr_key, 0))
            return

        fr_max = scf.N_frs[n2] if (n2 != -1) else len(xi1s)
        # simulate subsampling
        n1_step = 2 ** total_conv_stride_over_U1_realized
        for n1 in range(0, N_fr_padded, n1_step):
            # simulate unpadding
            if n1 / n1_step < ind_start_fr:
                continue
            elif n1 / n1_step >= ind_end_fr:
                break

            if n1 >= fr_max:  # equivalently `j1 >= j2`
                # these are padded rows, no associated filters
                xi1, sigma1, j1, is_cqt1 = nan, nan, nan, nan
            else:
                xi1, sigma1, j1, is_cqt1 = (xi1s[n1], sigma1s[n1], j1s[n1],
                                            is_cqt1s[n1])
            meta['order' ][pair].append(2)
            meta['xi'    ][pair].append((xi2,     xi1_fr,     xi1))
            meta['sigma' ][pair].append((sigma2,  sigma1_fr,  sigma1))
            meta['j'     ][pair].append((j2,      j1_fr,      j1))
            meta['is_cqt'][pair].append((is_cqt2, is_cqt1_fr, is_cqt1))
            meta['n'     ][pair].append((n2_n,    n1_fr_n,    n1))
            meta['s'     ][pair].append((spin,))
            meta['stride'][pair].append(stride)
            meta['key'   ][pair].append((n2_key,  n1_fr_key,  n1))

    # set params
    N, N_fr = 2**J_pad, 2**scf.J_pad_frs_max_init
    xi_min = (2 / N)  # leftmost peak at bin 2
    xi_min_fr = (2 / N_fr)
    log2_T = math.floor(math.log2(T))
    log2_F = math.floor(math.log2(F))
    # extract filter meta
    sigma_low, xi1s, sigma1s, j1s, is_cqt1s, xi2s, sigma2s, j2s, is_cqt2s = \
        calibrate_scattering_filters(J, Q, T, xi_min=xi_min, r_psi=r_psi)
    sigma_low_fr, xi1s_fr, sigma1s_fr, j1s_fr, is_cqt1_frs, *_ = \
        calibrate_scattering_filters(J_fr, Q_fr, F, xi_min=xi_min_fr,
                                     r_psi=scf.r_psi_fr)

    # compute modified meta if `resample_=False`
    sampling_psi_fr, sampling_phi_fr = sampling_filters_fr
    if sampling_psi_fr == 'recalibrate':
        (xi1s_fr_new, sigma1s_fr_new, j1s_fr_new, is_cqt1_frs_new, _
         ) = _recalibrate_psi_fr(xi1s_fr, sigma1s_fr, j1s_fr, is_cqt1_frs, N_fr,
                                 scf.alpha, scf.N_fr_scales_min,
                                 scf.N_fr_scales_max,
                                 scf.sigma_max_to_min_max_ratio)

    # fetch phi meta; must access `phi_f_fr` as `j1s_fr` requires sampling phi
    meta_phi = {}
    for field in ('xi', 'sigma', 'j'):
        meta_phi[field] = {}
        for k in scf.phi_f_fr[field]:
            meta_phi[field][k] = scf.phi_f_fr[field][k]
    xi1s_fr_phi, sigma1_fr_phi, j1s_fr_phi = list(meta_phi.values())

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
    for field in ('order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 's', 'stride',
                  'key'):
        meta[field] = {name: [] for name in coef_names}

    # Zeroth-order ###########################################################
    if average_global:
        k0 = log2_T
    elif average:
        k0 = max(log2_T - oversampling, 0)
    meta['order' ]['S0'].append(0)
    meta['xi'    ]['S0'].append((nan, nan, 0         if average else nan))
    meta['sigma' ]['S0'].append((nan, nan, sigma_low if average else nan))
    meta['j'     ]['S0'].append((nan, nan, log2_T    if average else nan))
    meta['is_cqt']['S0'].append((nan, nan, nan))
    meta['n'     ]['S0'].append((nan, nan, inf       if average else nan))
    meta['s'     ]['S0'].append((nan,))
    meta['stride']['S0'].append((nan, k0 if average else nan))
    meta['key'   ]['S0'].append((0, 0, 0))

    # First-order ############################################################
    def stride_S1(j1):
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)
        k1_J = max(log2_T - k1 - oversampling, 0)

        if average_global:
            total_conv_stride_tm = log2_T
        elif average:
            total_conv_stride_tm = k1 + k1_J
        else:
            total_conv_stride_tm = k1
        return total_conv_stride_tm

    for (n1, (xi1, sigma1, j1, is_cqt1)
         ) in enumerate(zip(xi1s, sigma1s, j1s, is_cqt1s)):
        meta['order' ]['S1'].append(1)
        meta['xi'    ]['S1'].append((nan, nan, xi1))
        meta['sigma' ]['S1'].append((nan, nan, sigma1))
        meta['j'     ]['S1'].append((nan, nan, j1))
        meta['is_cqt']['S1'].append((nan, nan, is_cqt1))
        meta['n'     ]['S1'].append((nan, nan, n1))
        meta['s'     ]['S1'].append((nan,))
        meta['stride']['S1'].append((nan, stride_S1(j1)))
        meta['key'   ]['S1'].append((0, 0, n1))

    S1_len = len(meta['n']['S1'])
    assert S1_len >= scf.N_frs_max

    # Joint scattering #######################################################
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

    array_fields = ['order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 's', 'stride',
                    'key']
    for field in array_fields:
        for pair, v in meta[field].items():
            meta[field][pair] = np.array(v)

    if out_3D:
      # reorder for 3D
      for field in array_fields:
        if field in ('s', 'order'):
            meta_len = 1
        elif field == 'stride':
            meta_len = 2
        else:
            meta_len = 3
        for pair in meta[field]:
          n_slices = None

          if pair in ('S0', 'S1'):
              # simply expand dim for consistency, no 3D structure
              meta[field][pair] = meta[field][pair].reshape(-1, 1, meta_len)
              continue

          elif 'psi_f' in pair:
              is_phi_t = pair.startswith('phi_t')
              if sampling_psi_fr != 'exclude':
                  number_of_n2 = (1 if is_phi_t else
                                  sum(j2 != 0 for j2 in j2s))
                  number_of_n1_fr = len(j1s_fr)
              else:
                  n_slices = 0
                  if is_phi_t:
                      n2 = -1
                      for n1_fr, j1_fr in enumerate(j1s_fr):
                          if _exclude_excess_scale(n2, n1_fr):
                              continue
                          n_slices += 1
                  else:
                      for n2, j2 in enumerate(j2s):
                          if j2 == 0:
                              continue
                          for n1_fr, j1_fr in enumerate(j1s_fr):
                              if _exclude_excess_scale(n2, n1_fr):
                                  continue
                              n_slices += 1

          elif pair == 'psi_t * phi_f':
              number_of_n2 = sum(j2 != 0 for j2 in j2s)
              number_of_n1_fr = 1

          elif pair == 'phi_t * phi_f':
              number_of_n2 = 1
              number_of_n1_fr = 1

          if n_slices is None:
              n_slices = number_of_n2 * number_of_n1_fr
          meta[field][pair] = meta[field][pair].reshape(n_slices, -1, meta_len)

    if out_exclude is not None:
        # drop excluded pairs
        for pair in out_exclude:
            for field in meta:
                del meta[field][pair]

    # ensure time / freq stride doesn't exceed log2_T / log2_F in averaged cases,
    # and J / J_fr in unaveraged
    smax_t_nophi = log2_T if average else J
    smax_f_nophi = log2_F if scf.average_fr else scf.J_fr
    for pair in meta['stride']:
        if pair == 'S0' and not average:
            continue
        smax_t = (smax_t_nophi if ('phi_t' not in pair) else
                  log2_T)
        smax_f = (smax_f_nophi if ('phi_f' not in pair) else
                  log2_F)
        for i, s in enumerate(meta['stride'][pair][..., 1].ravel()):
            assert s <= smax_t, ("meta['stride'][{}][{}] > stride_max_t "
                                 "({} > {})").format(pair, i, s, smax_t)
        if pair in ('S0', 'S1'):
            continue
        for i, s in enumerate(meta['stride'][pair][..., 0].ravel()):
            assert s <= smax_f, ("meta['stride'][{}][{}] > stride_max_f "
                                 "({} > {})").format(pair, i, s, smax_f)

    if not out_type.startswith('dict'):
        # join pairs
        if not out_3D:
            meta_flat = {f: np.concatenate([v for v in meta[f].values()], axis=0)
                         for f in meta}
        else:
            meta_flat0 = {f: np.concatenate(
                [v for k, v in meta[f].items() if k in ('S0', 'S1')],
                axis=0) for f in meta}
            meta_flat1 = {f: np.concatenate(
                [v for k, v in meta[f].items() if k not in ('S0', 'S1')],
                axis=0) for f in meta}
            meta_flat = (meta_flat0, meta_flat1)
        meta = meta_flat
    return meta
