# -*- coding: utf-8 -*-
"""Convenience utilities."""
import numpy as np
import scipy.signal
import warnings
from copy import deepcopy


def drop_batch_dim_jtfs(Scx):
    """Drop dim 0 of every JTFS coefficient (if it's `1`).

    Doesn't modify input:
        - dict/list: new list/dict (with copied meta if applicable)
        - array: new object but shared storage with original array (so original
          variable reference points to unsqueezed array).
    """
    def get_meta(s):
        return {k: v for k, v in s.items() if not hasattr(v, 'ndim')}

    backend = _infer_backend(Scx)
    if isinstance(Scx, dict):
        out = {}  # don't modify source dict
        for pair in Scx:
            out[pair] = []
            if isinstance(Scx[pair], list):
                for i, s in enumerate(Scx[pair]):
                    out[pair].append(get_meta(s))
                    out[pair][i]['coef'] = backend.squeeze(s['coef'], 0)
            else:
                out[pair] = backend.squeeze(Scx[pair], 0)
    elif isinstance(Scx, list):
        out = []  # don't modify source list
        for s in Scx:
            o = get_meta(s)
            o['coef'] = backend.squeeze(s['coef'], 0)
            out.append(o)
    elif isinstance(Scx, tuple):  # out_type=='array' && out_3D==True
        Scx = (backend.squeeze(Scx[0], 0),
               backend.squeeze(Scx[1], 0))
    elif hasattr(Scx, 'ndim'):
        Scx = backend.squeeze(Scx, 0)
    else:
        raise ValueError(("unrecognized input type: {}; must be as returned by "
                          "`jtfs(x)`.").format(type(Scx)))
    return out


def coeff_energy(Scx, meta, pair=None, aggregate=True, correction=False,
                 kind='l2'):
    """Computes energy of JTFS coefficients.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    pair: str / list/tuple[str] / None
        Name(s) of coefficient pairs whose energy to compute.
        If None, will compute for all.

    aggregate: bool (default True)
        True: return one value per `pair`, the sum of all its coeffs' energies
        False: return `(E_flat, E_slices)`, where:
            - E_flat = energy of every coefficient, flattened into a list
              (but still organized pair-wise)
            - E_slices = energy of every joint slice (if not `'S0', 'S1'`),
              in a pair. That is, sum of coeff energies on per-`(n2, n1_fr)`
              basis.

    correction : bool (default False)
        Whether to apply stride and filterbank norm correction factors:
            - stride: if subsampled by 2, energy will reduce by 2
            - filterbank: since input is assumed real, we convolve only over
              positive frequencies, getting half the energy

        Current JTFS implementation accounts for both so default is `False`.

        Filterbank energy correction is as follows:

            - S0 -> 1
            - U1 -> 2 (because psi_t is only analytic)
            - phi_t * phi_f -> 2 (because U1)
            - psi_t * phi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)
            - phi_t * psi_f -> 4 (because U1 and psi_f is only for one spin)
            - psi_t * psi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)

        For coefficient correction (e.g. distance computation) we instead
        scale the coefficients by square root of these values.

    kind: str['l1', 'l2']
        Kind of energy to compute. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)`
        (so actually L2^2).

    Returns
    -------
    E: float / tuple[list]
        Depends on `pair`, `aggregate`.

    Rationale
    ---------
    Simply `sum(abs(coef)**2)` won't work because we must account for subsampling.

        - Subsampling by `k` will reduce energy (both L1 and L2) by `k`
          (assuming no aliasing).
        - Must account for variable subsampling factors, both in time
          (if `average=False` for S0, S1) and frequency.
          This includes if only seeking ratio (e.g. up vs down), since
          `(a*x0 + b*x1) / (a*y0 + b*y1) != (x0 + x1) / (y0 + y1)`.
    """
    if pair is None or isinstance(pair, (tuple, list)):
        # compute for all (or multiple) pairs
        pairs = pair
        E_flat, E_slices = {}, {}
        for pair in Scx:
            if pairs is not None and pair not in pairs:
                continue
            E_flat[pair], E_slices[pair] = coeff_energy(
                Scx, meta, pair, aggregate=False, kind=kind)
        if aggregate:
            E = {}
            for pair in E_flat:
                E[pair] = np.sum(E_flat[pair])
            return E
        return E_flat, E_slices

    elif not isinstance(pair, str):
        raise ValueError("`pair` must be string, list/tuple of strings, or None "
                         "(got %s)" % pair)

    # compute compensation factor # TODO better comment
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: energy(c, kind=kind)
    norm_fn = lambda total_joint_stride: (2**total_joint_stride
                                          if correction else 1)

    E_flat, E_slices = _iterate_coeffs(Scx, meta, pair, fn, norm_fn, factor)
    Es = []
    for s in E_slices:
        Es.append(np.sum(s))
    E_slices = Es

    if aggregate:
        return np.sum(E_flat)
    return E_flat, E_slices


def coeff_distance(Scx0, Scx1, meta0, meta1=None, pair=None, correction=False,
                   kind='l2'):
    if meta1 is None:
        meta1 = meta0
    # compute compensation factor # TODO better comment
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: c

    def norm_fn(total_joint_stride):
        if not correction:
            return 1
        return (2**(total_joint_stride / 2) if kind == 'l2' else
                2**total_joint_stride)

    c_flat0, c_slices0 = _iterate_coeffs(Scx0, meta0, pair, fn, norm_fn, factor)
    c_flat1, c_slices1 = _iterate_coeffs(Scx1, meta0, pair, fn, norm_fn, factor)

    # make into array and assert shapes are as expected
    c_flat0, c_flat1 = np.asarray(c_flat0), np.asarray(c_flat1)
    c_slices0 = [np.asarray(c) for c in c_slices0]
    c_slices1 = [np.asarray(c) for c in c_slices1]

    assert c_flat0.ndim == c_flat1.ndim == 2, (c_flat0.shape, c_flat1.shape)
    shapes = [np.array(c).shape for cs in (c_slices0, c_slices1) for c in cs]
    # effectively 3D
    assert all(len(s) == 2 for s in shapes), shapes

    E = lambda x: _l2(x) if kind == 'l2' else _l1(x)
    ref0, ref1 = E(c_flat0), E(c_flat1)
    ref = (ref0 + ref1) / 2

    def D(x0, x1, axis):
        if isinstance(x0, list):
            return [D(_x0, _x1, axis) for (_x0, _x1) in zip(x0, x1)]

        if kind == 'l2':
            return np.sqrt(np.sum(np.abs(x0 - x1)**2, axis=axis)) / ref
        return np.sum(np.abs(x0 - x1), axis=axis) / ref

    # do not collapse `freq` dimension
    reldist_flat   = D(c_flat0,   c_flat1,   axis=-1)
    reldist_slices = D(c_slices0, c_slices1, axis=(-1, -2))

    # return tuple consistency; we don't do "slices" here
    return reldist_flat, reldist_slices


def _get_pair_factor(pair, correction):
    if pair == 'S0' or not correction:
        factor = 1
    elif 'psi' in pair:
        factor = 4
    else:
        factor = 2
    return factor


def _iterate_coeffs(Scx, meta, pair, fn=None, norm_fn=None, factor=None):
    coeffs = drop_batch_dim_jtfs(Scx)[pair]
    out_list = bool(isinstance(coeffs, list))

    # infer out_3D
    out_3D = bool(meta['stride'][pair].ndim == 3)
    if out_3D:
        assert coeffs.ndim == 3, coeffs.ndim
    # completely flatten into (*, time)
    if out_list:
        coeffs_flat = []
        for coef in coeffs:
            c = coef['coef']
            coeffs_flat.extend(c)
    else:
        if out_3D:
            coeffs = coeffs.reshape(-1, coeffs.shape[-1])
        coeffs_flat = coeffs

    # prepare for iterating
    meta = deepcopy(meta)  # don't change external dict
    if out_3D:
        meta['stride'][pair] = meta['stride'][pair].reshape(-1, 2)
        meta['n'][pair] = meta['n'][pair].reshape(-1, 3)

    assert (len(coeffs_flat) == len(meta['stride'][pair])), (
        "{} != {} | {}".format(len(coeffs_flat), len(meta['stride'][pair]), pair))

    # define helpers #########################################################
    def get_total_joint_stride(meta_idx):
        n_freqs = 1
        m_start, m_end = meta_idx[0], meta_idx[0] + n_freqs
        stride = meta['stride'][pair][m_start:m_end]
        assert len(stride) != 0, pair

        stride[np.isnan(stride)] = 0
        total_joint_stride = stride.sum()
        meta_idx[0] = m_end  # update idx
        return total_joint_stride

    def n_current():
        i = meta_idx[0]
        m = meta['n'][pair]
        return (m[i] if i <= len(m) - 1 else
                np.array([-3, -3]))  # reached end; ensure equality fails

    def n_is_equal(n0, n1):
        n0, n1 = n0[:2], n1[:2]  # discard U1
        n0[np.isnan(n0)], n1[np.isnan(n1)] = -2, -2  # NaN -> -2
        return bool(np.all(n0 == n1))

    # append energies one by one #############################################
    fn = fn or (lambda c: c)
    norm_fn = norm_fn or (lambda total_joint_stride: 2**total_joint_stride)
    factor = factor or 1

    E_flat = []
    E_slices = [] if pair in ('S0', 'S1') else [[]]
    meta_idx = [0]  # make mutable to avoid passing around
    for c in coeffs_flat:
        if hasattr(c, 'numpy'):
            if hasattr(c, 'cpu'):  # torch
                c = c.cpu()
            c = c.numpy()  # TF/torch
        n_prev = n_current()
        assert c.ndim == 1, (c.shape, pair)
        total_joint_stride = get_total_joint_stride(meta_idx)

        E = norm_fn(total_joint_stride) * fn(c) * factor
        E_flat.append(E)

        if pair in ('S0', 'S1'):
            E_slices.append(E)  # append to list of coeffs
        elif n_is_equal(n_current(), n_prev):
            E_slices[-1].append(E)  # append to slice
        else:
            E_slices[-1].append(E)  # append to slice
            E_slices.append([])

    # # in case loop terminates early
    if isinstance(E_slices[-1], list) and len(E_slices[-1]) == 0:
        E_slices.pop()

    # ensure they sum to same
    Es_sum = np.sum(np.sum(E_slices))
    adiff = abs(np.sum(E_flat) - Es_sum)
    assert np.allclose(np.sum(E_flat), Es_sum), "MAE=%.3f" % adiff

    return E_flat, E_slices


#### Validating 1D filterbank ################################################
def validate_filterbank_tm(sc=None, psi1_f=None, psi2_f=None, phi_f=None,
                           criterion_amplitude=1e-3, verbose=True):
    """Runs `validate_filterbank()` on temporal filters; supports `Scattering1D`
    and `TimeFrequencyScattering1D`.

    Parameters
    ----------
        sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
            If None, then `psi1_f_fr_up`, `psi1_f_fr_down`, and `phi_f_fr` must
            be not None.

        psi1_f : list[tensor] / None
            First-order bandpasses in frequency domain.
            Overridden if `sc` is not None.

        psi2_f : list[tensor] / None
            Second-order bandpasses in frequency domain.
            Overridden if `sc` is not None.

        phi_f : tensor / None
            Lowpass filter in frequency domain.
            Overridden if `sc` is not None.

        criterion_amplitude : float
            Used for various thresholding in `validate_filterbank()`.

        verbose : bool (default True)
            Whether to print the report.

    Returns
    -------
        data1, data2 : dict, dict
            Returns from `validate_filterbank()` for `psi1_f` and `psi2_f`.
    """
    if sc is None:
        assert not any(arg is None for arg in (psi1_f, psi2_f, phi_f))
    else:
        psi1_f, psi2_f, phi_f = [getattr(sc, k) for k in
                                 ('psi1_f', 'psi2_f', 'phi_f')]
    psi1_f, psi2_f = [[p[0] for p in ps] for ps in (psi1_f, psi2_f)]
    phi_f = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]

    if verbose:
        print("\n// FIRST-ORDER")
    data1 = validate_filterbank(psi1_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    if verbose:
        print("\n\n// SECOND-ORDER")
    data2 = validate_filterbank(psi2_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    return data1, data2


def validate_filterbank_fr(sc=None, psi1_f_fr_up=None, psi1_f_fr_down=None,
                           phi_f_fr=None, j0=0, criterion_amplitude=1e-3,
                           verbose=True):
    """Runs `validate_filterbank()` on frequential filters of JTFS.

    Parameters
    ----------
        sc : `TimeFrequencyScattering1D` / None
            JTFS instance. If None, then `psi1_f_fr_up`, `psi1_f_fr_down`, and
            `phi_f_fr` must be not None.

        psi1_f_fr_up : list[tensor] / None
            Spin up bandpasses in frequency domain.
            Overridden if `sc` is not None.

        psi1_f_fr_down : list[tensor] / None
            Spin down bandpasses in frequency domain.
            Overridden if `sc` is not None.

        phi_f_fr : tensor / None
            Lowpass filter in frequency domain.
            Overridden if `sc` is not None.

        j0 : int
            See `subsample_equiv_due_to_pad` in `help(TimeFrequencyScattering1D)`.

        criterion_amplitude : float
            Used for various thresholding in `validate_filterbank()`.

        verbose : bool (default True)
            Whether to print the report.

    Returns
    -------
        data_up, data_down : dict, dict
            Returns from `validate_filterbank()` for `psi1_f_fr_up` and
            `psi1_f_fr_down`.
    """
    if sc is None:
        assert not any(arg is None for arg in
                       (psi1_f_fr_up, psi1_f_fr_down, phi_f_fr))
    else:
        psi1_f_fr_up, psi1_f_fr_down, phi_f_fr = [
            getattr(sc, k) for k in
            ('psi1_f_fr_up', 'psi1_f_fr_down', 'phi_f_fr')]
    psi1_f_fr_up, psi1_f_fr_down = [[p[j0] for p in ps] for ps in
                                    (psi1_f_fr_up, psi1_f_fr_down)]
    phi_f_fr = phi_f_fr[j0][0]

    if verbose:
        print("\n// SPIN UP")
    data_up = validate_filterbank(psi1_f_fr_up, phi_f_fr, criterion_amplitude,
                                  verbose=verbose,
                                  for_real_inputs=False, unimodal=True)
    if verbose:
        print("\n\n// SPIN DOWN")
    data_down = validate_filterbank(psi1_f_fr_down, phi_f_fr, criterion_amplitude,
                                    verbose=verbose,
                                    for_real_inputs=False, unimodal=True)
    return data_up, data_down


def validate_filterbank(psi_fs, phi_f=None, criterion_amplitude=1e-3,
                        for_real_inputs=True, unimodal=True, verbose=True):
    """Checks whether the wavelet filterbank is well-behaved against several
    criterion:

        1. Analyticity:
          - A: Whether analytic *and* anti-analytic filters are present
               (input should contain only one)
          - B: Extent of (anti-)analyticity - whether there's components
               on other side of Nyquist
        2. Aliasing:
          - A. Whether peaks are sorted (left to right or right to left).
               If not, it's possible aliasing (or sloppy user input).
          - B. Whether peaks are distributed exponentially or linearly.
               If neither, it's possible aliasing. (Detection isn't foulproof.)
        3. Zero-mean: whether filters are zero-mean (in time domain)
        4. Zero-phase: whether filters are zero-phase
        5. Frequency coverage: whether filters capture every frequency,
           and whether they do so excessively or insufficiently.
             - Measured with Littlewood-Paley sum (sum of energies).
        6. Redundancy: whether filters overlap excessively (this isn't
           necessarily bad).
             - Measured as ratio of product of energies to sum of energies
               of adjacent filters
        7. Decay:
          - A: Whether any filter is a pure sine (occurs if we try to sample
               a wavelet at too low of a center frequency)
          - B: Whether filters decay sufficiently (in time domain) to avoid
               boundary effects
          B may fail for same reason as 8A & 8B (see these).
        8. Temporal peaks:
          - A: Whether peak is at t==0
          - B: Whether there is only one peak
          - C: Whether decay is smooth (else will incur inflection points)
          A and B may fail to hold for lowest xi due to Morlet's corrective
          term; this is proper behavior.
          See https://www.desmos.com/calculator/3ycdx1n9j6

    Parameters
    ----------
        psi_fs : list[tensor]
            Wavelets in frequency domain, of same length.

        phi_f : tensor
            Lowpass filter in frequency domain, of same length as `psi_fs`.

        criterion_amplitude : float
            Used for various thresholding.

        for_real_inputs : bool (default True)
            Whether the filterbank is intended for real-only inputs.
            E.g. `False` for spinned bandpasses in JTFS.

        unimodal : bool (default True)
            Whether the wavelets have a single peak in frequency domain.
            If `False`, some checks are omitted, and others might be inaccurate.
            Always `True` for Morlet wavelets.

        verbose : bool (default True)
            Whether to print the report.

    Returns
    -------
        data : dict
            Aggregated testing info, along with the report. For keys, see
            `print(list(data))`.
    """
    def pop_if_no_header(report, did_header):
        if not did_header:
            report.pop(-1)

    # fetch basic metadata
    psi_f_0 = psi_fs[0]  # reference filter
    N = len(psi_f_0)

    # assert all inputs are same length
    for n, p in enumerate(psi_fs):
        assert len(p) == N, (len(p), N)
    if phi_f is not None:
        assert len(phi_f) == N, (len(phi_f), N)
    # squeeze into 1D for convenience later
    psi_fs = [p.squeeze() for p in psi_fs]
    if phi_f is not None:
        phi_f = phi_f.squeeze()

    # initialize report
    report = []
    data = {k: {} for k in ('analytic_a_ratio', 'nonzero_mean', 'sine', 'decay',
                            'imag_mean', 'time_peak_idx', 'n_inflections',
                            'redundancy')}
    data['opposite_analytic'] = []

    def title(txt):
        return ("\n== {} " + "=" * (80 - len(txt)) + "\n").format(txt)
    # for later
    # w_pos = np.linspace(0, .5, N//2 + 1, endpoint=True)  # TODO
    w_pos = np.linspace(0, N//2, N//2 + 1, endpoint=True).astype(int)
    w_neg = - w_pos[1:-1][::-1]
    w = np.hstack([w_pos, w_neg])
    eps = np.finfo(psi_f_0.dtype).eps

    peak_idxs = np.array([np.argmax(np.abs(p)) for p in psi_fs])
    peak_idxs_sorted = np.sort(peak_idxs)
    if unimodal and not (np.all(peak_idxs == peak_idxs_sorted) or
                         np.all(peak_idxs == peak_idxs_sorted[::-1])):
        warnings.warn("`psi_fs` peak locations are not sorted; a possible reason "
                      "is aliasing. Will sort, breaking mapping with input's.")
        data['not_sorted'] = True
        peak_idxs = peak_idxs_sorted

    # Analyticity ############################################################
    # check if there are both analytic and anti-analytic bandpasses ##########
    report += [title("ANALYTICITY")]
    did_header = False

    peak_idx_0 = np.argmax(psi_f_0)
    if peak_idx_0 == N // 2:  # ambiguous case; check next filter
        peak_idx_0 = np.argmax(psi_fs[1])
    analytic_0 = bool(peak_idx_0 < N//2)
    # assume entire filterbank is per psi_0
    analyticity = "analytic" if analytic_0 else "anti-analytic"

    for n, p in enumerate(psi_fs[1:]):
        peak_idx_n = np.argmax(p)
        analytic_n = bool(peak_idx_n < N//2)
        if not (analytic_0 is analytic_n):
            if not did_header:
                report += [("Found analytic AND anti-analytic filters in same "
                            "filterbank! psi_fs[0] is {}, but the following "
                            "aren't:\n").format(analyticity)]
                did_header = True
            report += [f"psi_fs[{n}]\n"]
            data['opposite_analytic'].append(n)

    # check if any bandpass isn't strictly analytic/anti- ####################
    did_header = False
    th_ratio = (1 / criterion_amplitude)
    for n, p in enumerate(psi_fs):
        ap = np.abs(p)
        # assume entire filterbank is per psi_0
        if analytic_0:
            # Nyquist is *at* N//2, so to include in sum, index up to N//2 + 1
            a_ratio = (ap[:N//2 + 1].sum() / (ap[N//2 + 1:].sum() + eps))
        else:
            a_ratio = (ap[N//2:].sum() / (ap[:N//2].sum() + eps))
        if a_ratio < th_ratio:
            if not did_header:
                report += [("Found not strictly {} filter(s); threshold for "
                            "ratio of `spectral sum` to `spectral sum past "
                            "Nyquist` is {} - got (less is worse):\n"
                            ).format(analyticity, th_ratio)]
                did_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, a_ratio)]
            data['analytic_a_ratio'][n] = a_ratio

    # check if any bandpass isn't zero-mean ##################################
    pop_if_no_header(report, did_header)
    report += [title("ZERO-MEAN")]
    did_header = False

    for n, p in enumerate(psi_fs):
        if p[0] != 0:
            if not did_header:
                report += ["Found non-zero mean filter(s)!:\n"]
                did_header = True
            report += ["psi_fs[{}][0] == {:.2e}\n".format(n, p[0])]
            data['nonzero_mean'][n] = p[0]

    # Littlewood-Paley sum ###################################################
    def report_lp_sum(report, phi):
        with_phi = not isinstance(phi, int)
        s = "with" if with_phi else "without"
        report += [title("LP-SUM (%s phi)" % s)]
        did_header = False

        lp_sum = lp_sum_psi + np.abs(phi)**2
        lp_sum = (lp_sum[:N//2 + 1] if analytic_0 else
                  lp_sum[N//2:])
        if with_phi:
            data['lp'] = lp_sum
        else:
            data['lp_no_phi'] = lp_sum
        if not with_phi and analytic_0:
            lp_sum = lp_sum[1:]  # exclude dc

        diff_over  = lp_sum - th_lp_sum_over
        diff_under = th_lp_sum_under - lp_sum
        diff_over_max, diff_under_max = diff_over.max(), diff_under.max()
        excess_over  = np.where(diff_over  > th_sum_excess)[0]
        excess_under = np.where(diff_under > th_sum_excess)[0]
        if not analytic_0:
            excess_over  += N//2
            excess_under += N//2
        elif analytic_0 and not with_phi:
            excess_over += 1
            excess_under += 1  # dc

        input_kind = "real" if for_real_inputs else "complex"
        if len(excess_over) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_over) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum exceeds threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_over, input_kind, diff_over_max,
                                 N//2, s)]
            # report += ["{}\n\n".format(np.round(w[excess_over][::stride], 3))]
            report += ["{}\n\n".format(w[excess_over][::stride])]
            did_header = True
            if with_phi:
                data['lp_excess_over'] = excess_over
                data['lp_excess_over_max'] = diff_over_max
            else:
                data['lp_no_phi_excess_over'] = excess_over
                data['lp_no_phi_excess_over_max'] = diff_over_max

        if len(excess_under) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_under) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum falls below threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse; ~{} implies ~zero "
                        "capturing of the frequency!) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_under, input_kind, diff_under_max,
                                 th_lp_sum_under, N//2, s)]
            # w_show = np.round(w[excess_under][::stride], 3)
            report += ["{}\n\n".format(w[excess_under][::stride])]
            did_header = True
            if with_phi:
                data['lp_excess_under'] = excess_under
                data['lp_excess_under_max'] = diff_under_max
            else:
                data['lp_no_phi_excess_under'] = excess_under
                data['lp_no_phi_excess_under_max'] = diff_under_max

        if did_header:
            stdev = np.abs(lp_sum[lp_sum >= th_lp_sum_under] -
                           th_lp_sum_under).std()
            report += [("Mean absolute deviation from tight frame: {:.2f}\n"
                        "Standard deviation from tight frame: {:.2f} "
                        "(excluded LP sum values below {})\n").format(
                            np.abs(diff_over).mean(), stdev, th_lp_sum_under)]

        pop_if_no_header(report, did_header)

    pop_if_no_header(report, did_header)
    th_lp_sum_over = 2 if for_real_inputs else 1
    th_lp_sum_under = th_lp_sum_over / 2
    th_sum_excess = (1 + criterion_amplitude)**2 - 1
    lp_sum_psi = np.sum([np.abs(p)**2 for p in psi_fs], axis=0)

    # do both cases
    if phi_f is not None:
        report_lp_sum(report, phi=phi_f)
    report_lp_sum(report, phi=0)

    # Redundancy #############################################################
    from .scattering1d.filter_bank import compute_filter_redundancy

    report += [title("REDUNDANCY")]
    did_header = False
    th_r = .4 if for_real_inputs else .2

    max_to_print = 20
    printed = 0
    for n in range(len(psi_fs) - 1):
        r = compute_filter_redundancy(psi_fs[n], psi_fs[n + 1])
        data['redundancy'][(n, n + 1)] = r
        if r > th_r:
            if not did_header:
                report += [("Found filters with redundancy exceeding {} (energy "
                            "overlap relative to sum of individual energies) "
                            "-- This isn't necessarily bad. Showing up to 20 "
                            "filters:\n").format(th_r)]
                did_header = True
            report += ["psi_fs[{}] & psi_fs[{}]: {:.3f}\n".format(n, n + 1, r)]
            printed += 1
            if printed >= max_to_print:
                break

    # Decay: check if any bandpass is a pure sine ############################
    pop_if_no_header(report, did_header)
    report += [title("DECAY (check for pure sines)")]
    did_header = False
    th_ratio_max_to_next_max = (1 / criterion_amplitude)

    for n, p in enumerate(psi_fs):
        psort = np.sort(np.abs(p))  # least to greatest
        ratio = psort[-1] / (psort[-2] + eps)
        if ratio > th_ratio_max_to_next_max:
            if not did_header:
                report += [("Found filter(s) that are pure sines! Threshold for "
                            "ratio of Fourier peak to next-highest value is {} "
                            "- got (more is worse):\n"
                            ).format(th_ratio_max_to_next_max)]
                did_header = True
            report += ["psi_fs[{}]: {:.2e}\n".format(n, ratio)]
            data['sine'][n] = ratio

    # Decay: boundary effects ################################################
    pop_if_no_header(report, did_header)
    report += [title("DECAY (boundary effects)")]
    did_header = False
    th_ratio_max_to_min = (1 / criterion_amplitude)

    psis = [np.fft.ifft(p) for p in psi_fs]
    apsis = [np.abs(p) for p in psis]
    for n, ap in enumerate(apsis):
        ratio = ap.max() / (ap.min() + eps)
        if ratio < th_ratio_max_to_min:
            if not did_header:
                report += [("Found filter(s) with incomplete decay (will incur "
                            "boundary effects), with following ratios of "
                            "amplitude max to edge (less is worse; threshold "
                            "is {}):\n").format(1 / criterion_amplitude)]
                did_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, ratio)]
            data['decay'][n] = ratio

    # check lowpass
    if phi_f is not None:
        aphi = np.abs(np.fft.ifft(phi_f))
        ratio = aphi.max() / (aphi.min() + eps)
        if ratio < th_ratio_max_to_min:
            nl = "\n" if did_header else ""
            report += [("{}Lowpass filter has incomplete decay (will incur "
                        "boundary effects), with following ratio of amplitude "
                        "max to edge: {:.1f} > {}\n").format(nl, ratio,
                                                             th_ratio_max_to_min)]
            did_header = True
            data['decay'][-1] = ratio

    # Phase ##################################################################
    pop_if_no_header(report, did_header)
    report += [title("PHASE")]
    did_header = False
    th_imag_mean = eps

    for n, p in enumerate(psi_fs):
        imag_mean = np.abs(p.imag).mean()
        if imag_mean > th_imag_mean:
            if not did_header:
                report += [("Found filters with non-zero phase, with following "
                            "absolute mean imaginary values:\n")]
            report += ["psi_fs[{}]: {:.1e}\n".format(n, imag_mean)]
            data['imag_mean'][n] = imag_mean

    # Aliasing ###############################################################
    def diff_extend(diff, th, cond='gt', order=1):
        # the idea is to take `diff` without losing samples, if the goal is
        # `where(diff == 0)`; `diff` is forward difference, and two samples
        # participated in producing the zero, where later one's index is dropped
        # E.g. detecting duplicate peak indices:
        # [0, 1, 3, 3, 5] -> diff gives [2], so take [2, 3]
        # but instead of adding an index, replace next sample with zero such that
        # its `where == 0` produces that index
        if order > 1:
            diff_e = diff_extend(diff, th)
            for o in range(order - 1):
                diff_e = diff_e(diff_e, th)
            return diff_e

        diff_e = []
        d_extend = 2*th if cond == 'gt' else th
        prev_true = False
        for d in diff:
            if prev_true:
                diff_e.append(d_extend)
                prev_true = False
            else:
                diff_e.append(d)
            if (cond == 'gt' and np.abs(d) > th or
                cond == 'eq' and np.abs(d) == th):
                prev_true = True
        if prev_true:
            # last sample was zero; extend
            diff_e.append(d_extend)
        return np.array(diff_e)

    if unimodal:
        pop_if_no_header(report, did_header)
        report += [title("ALIASING")]
        did_header = False
        eps_big = eps * 100  # ease threshold for "zero"

        # check whether peak locations follow a linear or exponential
        # distribution, progressively dropping those that do to see if any remain

        # x[n] = A^n + C; x[n] - x[n - 1] = A^n - A^(n-1) = A^n*(1 - A) = A^n*C
        # log(A^n*C) = K + n; diff(diff(K + n)) == 0
        # `abs` for anti-analytic case with descending idxs
        logdiffs = np.diff(np.log(np.abs(np.diff(peak_idxs))), 2)
        # In general it's impossible to determine whether a rounded sequence
        # samples an exponential, since if the exponential rate (A in A^n) is
        # sufficiently small, its rounded values will be linear over some portion.
        # However, it cannot be anything else, and we are okay with linear
        # (unless constant, i.e. duplicate, captured elsewhere) - thus the net
        # case of `exp + lin` is still captured. The only uncertainty is in
        # the threshold; assuming deviation by at most 1 sample, we set it to 1.
        # A test is:
        # `for b in linspace(1.2, 6.5, 500): x = round(b**arange(10) + 50)`
        # with `if any(abs(diff, o).min() == 0 for o in (1, 2, 3)): continue`,
        # Another with: `linspace(.2, 1, 500)` and `round(256*b**arange(10) + 50)`
        # to exclude `x` with repeated or linear values
        # However, while this has no false positives (never misses an exp/lin),
        # it can also count some non-exp/lin as exp/lin, but this is rare.
        # To be safe, per above test, we use the empirical value of 0.9
        logdiffs_extended = diff_extend(logdiffs, .9)
        if len(logdiffs_extended) > len(logdiffs) + 2:
            # this could be `assert` but not worth erroring over this
            warnings.warn("`len(logdiffs_extended) > len(logdiffs) + 2`; will "
                          "use more conservative estimate on peaks distribution")
            logdiffs_extended = logdiffs
        keep = np.where(np.abs(logdiffs_extended) > .9)
        # note due to three `diff`s we artificially exclude 3 samples
        peak_idxs_remainder = peak_idxs[keep]

        # now constant (diff_order==1) and linear (diff_order==2)
        for diff_order in (1, 2):
            idxs_diff2 = np.diff(peak_idxs_remainder, diff_order)
            keep = np.where(np.abs(idxs_diff2) > eps_big)
            peak_idxs_remainder = peak_idxs_remainder[keep]

        # if anything remains, it's neither
        if len(peak_idxs_remainder) > 0:
            report += [("Found Fourier peaks that are spaced neither "
                        "exponentially nor linearly, suggesting possible "
                        "aliasing.\npsi_fs[n], n={}\n"
                        ).format(peak_idxs_remainder)]
            data['alias_peak_idxs'] = peak_idxs_remainder
            did_header = True

        # check if any peaks repeat ##########################################
        # e.g. [0, 1, 5, 5, 3] -> diff gives [3], so take [3, 4]
        peak_idxs_zeros = diff_extend(np.diff(peak_idxs), th=0, cond='eq')
        peak_idxs_zeros = np.where(peak_idxs_zeros == 0)[0]
        if len(peak_idxs_zeros) > 0:
            nl = "\n" if did_header else ""
            report += ["{}Found duplicate Fourier peaks!:\n{}\n".format(
                nl, peak_idxs_zeros)]
            did_header = True

    # Temporal peak ##########################################################
    if unimodal:
        # check that temporal peak is at t==0 ################################
        pop_if_no_header(report, did_header)
        report += [title("TEMPORAL PEAK")]
        did_header = False

        for n, ap in enumerate(apsis):
            peak_idx = np.argmax(ap)
            if peak_idx != 0:
                if not did_header:
                    report += [("Found filters with temporal peak not at t=0!, "
                                "with following peak locations:\n")]
                    did_header = True
                report += ["psi_fs[{}]: {}\n".format(n, peak_idx)]
                data['time_peak_idx'][n] = peak_idx

        # check that there is only one temporal peak #########################
        pop_if_no_header(report, did_header)
        did_header = False
        for n, ap in enumerate(apsis):
            # count number of inflection points (where sign of derivative changes)
            # exclude very small values
            # center for proper `diff`
            ap = np.fft.ifftshift(ap)
            inflections = np.diff(np.sign(np.diff(ap[ap > 10*eps])))
            n_inflections = sum(np.abs(inflections) > eps)

            if n_inflections > 1:
                if not did_header:
                    report += [("\nFound filters with multiple temporal peaks "
                                "(or incomplete/non-smooth decay)! "
                                "(more precisely, >1 inflection points) with "
                                "following number of inflection points:\n")]
                    did_header = True
                report += ["psi_fs[{}]: {}\n".format(n, n_inflections)]
                data['n_inflections'] = n_inflections

    # Print report ###########################################################
    pop_if_no_header(report, did_header)
    report = ''.join(report)
    data['report'] = report
    if verbose:
        if len(report) == 0:
            print("Perfect filterbank!")
        else:
            print(report)
    return data


#### energy & distance #######################################################
def energy(x, kind='l2'):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    def sum(x):
        if type(x).__module__ == 'tensorflow':
            return backend.reduce_sum(x)
        return backend.sum(x)

    x = x['coef'] if isinstance(x, dict) else x
    backend = _infer_backend(x)
    return (backend.sum(backend.abs(x)) if kind == 'l1' else
            backend.sum(backend.abs(x)**2))


def _l2(x):  # TODO backend # TODO use `energy` instead?
    return np.sqrt(np.sum(np.abs(x)**2))

def l2(x0, x1, adj=False):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    ref = _l2(x0) if not adj else (_l2(x0) + _l2(x1)) / 2
    return _l2(x1 - x0) / ref


def _l1(x):
    return np.sum(np.abs(x))

def l1(x0, x1, adj=False):
    ref = _l1(x0) if not adj else (_l1(x0) + _l1(x1)) / 2
    return _l1(x1 - x0) / ref


def rel_ae(x0, x1, eps=None, ref_both=True):
    """Relative absolute error."""
    if ref_both:
        if eps is None:
            eps = (np.abs(x0).max() + np.abs(x1).max()) / 2000
        ref = (x0 + x1)/2 + eps
    else:
        if eps is None:
            eps = np.abs(x0).max() / 1000
        ref = x0 + eps
    return np.abs(x0 - x1) / ref


#### test signals ###########################################################
def echirp(N, fmin=.1, fmax=None, tmin=0, tmax=1):
    """https://overlordgolddragon.github.io/test-signals/ (bottom)"""
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)

    phi = 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return np.cos(phi)


def fdts(N, n_partials=2, total_shift=None, f0=None, seg_len=None,
         endpoint=False):
    """Generate windowed tones with Frequency-dependent Time Shifts (FDTS)."""
    total_shift = total_shift or N//16
    f0 = f0 or N//12
    seg_len = seg_len or N//8

    t = np.linspace(0, 1, N, endpoint=endpoint)
    window = scipy.signal.tukey(seg_len, alpha=0.5)
    pad_right = (N - len(window)) // 2
    pad_left = N - len(window) - pad_right
    window = np.pad(window, [pad_left, pad_right])

    x = np.zeros(N)
    xs = x.copy()
    for p in range(1, 1 + n_partials):
        x_partial = np.sin(2*np.pi * p*f0 * t) * window
        partial_shift = int(total_shift * np.log2(p) / np.log2(n_partials)
                            ) - total_shift//2
        xs_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        xs += xs_partial
    return x, xs

#### misc ###################################################################
def _infer_backend(x, get_name=False):
    while isinstance(x, (dict, list)):
        if isinstance(x, dict):
            if 'coef' in x:
                x = x['coef']
            else:
                x = list(x.values())[0]
        else:
            x = x[0]
    module = type(x).__module__
    if module == 'numpy':
        backend = np
    elif module == 'torch':
        import torch
        backend = torch
    elif module == 'tensorflow':
        import tensorflow
        backend = tensorflow
    else:
        raise ValueError("could not infer backend from %s" % type(x))
    return (backend if not get_name else
            (backend, module))
