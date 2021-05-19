
def timefrequency_scattering(
        x, pad, unpad, backend, J, log2_T, psi1, psi2, phi, sc_freq,
        pad_left=0, pad_right=0, ind_start=None, ind_end=None,
        oversampling=0, oversampling_fr=0, aligned=True, average=True,
        average_global=None, out_type='array', out_3D=False, pad_mode='zero'):
    """
    Main function implementing the Joint Time-Frequency Scattering transform.

    Below is implementation documentation for developers.

    Frequential scattering
    ======================

    Variables
    ---------
    Explanation of variable naming and roles. For `sc_freq`'s attributes see
    full Attributes docs in `TimeFrequencyScattering1D`.

    subsample_equiv_due_to_pad
        Equivalent amount of subsampling due to padding, relative to
        `J_pad_fr_max_init`.
        Distinguishes differences in input lengths (to `_frequency_scattering()`
        and `_joint_lowpass()`) due to padding and conv subsampling.
        This is needed to tell whether input is *trimmed* (pad difference)
        or *subsampled*.

    "conv subsampling"
        Refers to `subsample_fourier()` after convolution (as opposed to
        equivalent subsampling due to padding). i.e. "actual" subsampling.
        Refers to `total_convolutional_stride_over_U1`.

    total_convolutional_stride_over_U1
        See above. Alt name: `total_convolutional_stride_fr`; `over_U1` seeks
        to emphasize that it's the absolute stride over first order coefficients
        that matters rather than equivalent/relative quantities w.r.t. padded etc

    reference_subsample_equiv_due_to_pad
        If `aligned=True`, we must subsample as we did in minimally padded case;
        this equals `subsample_equiv_due_to_pad` in minimally padded case, and
        controls subsequent subsamplings. This is needed to preserve alignment
        of (first order) frequency rows by making conv stride the same for all
        input lengths.

    n1_fr_subsample
        Amount of conv subsampling done in `_frequency_scattering()`.

    reference_total_subsample_so_far
        `_joint_lowpass()`'s version of `reference_subsample_equiv_due_to_pad`,
        additionally accounting for `n1_fr_subsample` to determine amount of
        subsampling after lowpass.

    total_subsample_fr
        Total amount of subsampling, conv and equivalent, relative to
        `J_pad_fr_max_init`. Controls fr unpadding.

    Subsampling, padding
    --------------------
    Controlled by `aligned`, `out_3D`, `average_fr`, `log2_F`, and
    `resample_psi_fr` & `resample_phi_fr`.

    - `freq` == number of frequential rows (originating from U1), e.g. as in
      `(n1_fr, freq, time)` for joint slice shapes per `n2` (with `out_3D=True`).
    - `n1_fr` == number of frequential joint slices, or `psi1_f_fr_*`, per `n2`
    - `n2` == number of `psi2` wavelets, together with `n1_fr` controlling
      total number of joint slices

    aligned=True:  # TODO do these hold for `* phi_f` pairs?
        Imposes:
          - `total_convolutional_stride_over_U1` to be same for all joint
            coefficients. Otherwise, row-to-row log-frequency differences,
            `dw,2*dw,...`, will vary across joint slices, which breaks alignment.
          - `resample_psi_fr==True`: center frequencies must be same
          - `resample_phi_fr`: not necessarily restricted, as alignment is
            preserved under different amounts of frequential smoothing, but
            bins may blur together (rather unblur since `False` is finer);
            for same fineness/coarseness across slices, `True` is required.
            - greater `log2_F` won't necessarily yield greater conv stride,
              via `reference_total_subsample_so_far`

        average_fr=False:
            Additionally imposes that `total_convolutional_stride_over_U1==0`,
            since `psi_fr[0]['j'][0] == 0` and `_joint_lowpass()` can no longer
            compensate for variable `j_fr`.

        out_3D=True:
            Additionally imposes that all frequential padding is the same
            (maximum), since otherwise same `total_convolutional_stride_over_U1`
            yields variable `freq` across `n2`.

        average_fr_global=True:
            Stride logic in `_joint_lowpass()` is relaxed since all U1 are
            collapsed into one point; will no longer check
            `total_convolutional_stride_over_U1`  # TODO makes sense?

    out_3D=True:
        Imposes:
            - `freq`  to be same for all `n2` (can differ due to padding
              or convolutional stride)
            - `n1_fr` to be same for all `n2` -> `sampling_psi_fr != 'exclude'`

    log2_F:
        Larger -> smaller `freq`
        Larger -> greater `max_subsampling_before_phi_fr`
        Larger -> greater `J_pad_fr_max` (only if `log2_F > J_fr`)
        # TODO ^ check this against `prev_psi_halfwidth == prev_psi.size // 2`
    """
    # pack for later
    B = backend
    average_fr = sc_freq.average_fr
    commons = (B, sc_freq, aligned, oversampling_fr, average_fr, oversampling,
               average, unpad, log2_T, phi, ind_start, ind_end, out_3D)

    out_S_0 = []
    out_S_1 = []
    out_S_2 = {'psi_t * psi_f': [[], []],
               'psi_t * phi_f': [],
               'phi_t * psi_f': [[]],
               'phi_t * phi_f': []}

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right, pad_mode=pad_mode)
    # compute the Fourier transform
    U_0_hat = B.rfft(U_0)

    # Zeroth order ###########################################################
    if average:
        k0 = max(log2_T - oversampling, 0)
        S_0_c = B.cdgmm(U_0_hat, phi[0])
        S_0_hat = B.subsample_fourier(S_0_c, 2**k0)
        S_0_r = B.irfft(S_0_hat)
        S_0 = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x
    out_S_0.append({'coef': S_0,
                    'j': (log2_T,) if average else (),
                    'n': (-1,) if average else (),
                    's': ()})

    # First order ############################################################
    U_1_hat_list, S_1_list, S_1_c_list = [], [], []
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)
        U_1_c = B.cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = B.subsample_fourier(U_1_c, 2**k1)
        U_1_c = B.ifft(U_1_hat)

        # Modulus
        U_1_m = B.modulus(U_1_c)

        # Map to Fourier domain
        U_1_hat = B.rfft(U_1_m)
        U_1_hat_list.append(U_1_hat)

        # compute even if `average=False`, since used in `phi_t * psi_f` pairs
        S_1_c = B.cdgmm(U_1_hat, phi[k1])
        S_1_c_list.append(S_1_c)

        # Apply low-pass filtering over time (optional) and unpad
        if average_global:
            S_1 = B.mean(U_1_m, axis=-1)
        elif average:
            # Low-pass filtering over time
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_hat = B.subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = B.irfft(S_1_hat)
            # Unpad
            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            # Unpad
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])
        S_1_list.append(S_1)
        out_S_1.append({'coef': S_1, 'j': (j1,), 'n': (n1,), 's': ()})

    # Frequential averaging over time averaged coefficients ##################
    # `U1 * (phi_t * phi_f)` pair

    # zero-pad along frequency, map to Fourier domain
    pad_fr = sc_freq.J_pad_fr_max
    S_1_fr = _right_pad(S_1_list, pad_fr, B)
    # S_1_fr = B.zeros_like(S_1_list[-1], (2**pad_fr, S_1_list[-1].shape[-1]))
    # S_1_fr[:len(S_1_list)] = S_1_list

    if sc_freq.average_fr_global:
        S_1_fr = B.mean(S_1_fr, axis=-2)  # TODO axis will change
    else:
        S_1_tm_T_hat = _transpose_fft(S_1_fr, B, B.rfft)

        # this is usually 0
        subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr
        total_subsample_so_far = subsample_equiv_due_to_pad
        n1_fr_subsample = 0  # no intermediate scattering

        # TODO duplication?
        if aligned:
            # subsample as we would in min-padded case
            reference_subsample_equiv_due_to_pad = max(
                sc_freq.subsampling_equiv_relative_to_max_padding)
            if out_3D:
                # this is usually 0
                subsample_equiv_due_to_pad_min = (sc_freq.J_pad_fr_max_init -
                                                  sc_freq.J_pad_fr_max)
            else:
                subsample_equiv_due_to_pad_min = (
                    reference_subsample_equiv_due_to_pad)
            reference_total_subsample_so_far = (subsample_equiv_due_to_pad_min +
                                                n1_fr_subsample)
        else:
            # subsample regularly (relative to current padding)
            reference_total_subsample_so_far = total_subsample_so_far
        total_subsample_fr_max = sc_freq.log2_F
        lowpass_subsample_fr = max(total_subsample_fr_max -
                                   reference_total_subsample_so_far -
                                   oversampling_fr, 0)
        total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr

        # Low-pass filtering over frequency
        phi_fr = sc_freq.phi_f_fr[subsample_equiv_due_to_pad]
        S_1_fr_T_c = B.cdgmm(S_1_tm_T_hat, phi_fr)
        S_1_fr_T_hat = B.subsample_fourier(S_1_fr_T_c, 2**lowpass_subsample_fr)
        S_1_fr_T = B.irfft(S_1_fr_T_hat)

        # unpad + transpose, append to out
        if out_3D:
            S_1_fr_T = unpad(S_1_fr_T,
                             sc_freq.ind_start_fr_max[total_subsample_fr],
                             sc_freq.ind_end_fr_max[total_subsample_fr])
        else:
            S_1_fr_T = unpad(S_1_fr_T,
                             sc_freq.ind_start_fr[-1][total_subsample_fr],
                             sc_freq.ind_end_fr[-1][total_subsample_fr])
        S_1_fr = B.transpose(S_1_fr_T)
    out_S_2['phi_t * phi_f'].append({'coef': S_1_fr,
                                     'j': (log2_T, sc_freq.log2_F),
                                     'n': (-1, -1),
                                     's': (0,)})
    # set reference for later
    sc_freq.__total_convolutional_stride_over_U1 = (n1_fr_subsample +
                                                    lowpass_subsample_fr)

    ##########################################################################
    # Joint scattering: separable convolutions (along time & freq), and low-pass
    # `U1 * (psi_t * psi_f)` (up & down), and `U1 * (psi_t * phi_f)`
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        if j2 == 0:
            continue

        # preallocate output slice
        if aligned and out_3D:
            pad_fr = sc_freq.J_pad_fr_max
        else:
            pad_fr = sc_freq.J_pad_fr[n2]
        sub2_adj = min(j2, log2_T) if average else j2
        n2_time = U_0.shape[-1] // 2**max(sub2_adj - oversampling, 0)
        # Y_2_arr = B.zeros_like(U_1_c, (2**pad_fr, n2_time))
        Y_2_list = []

        # Wavelet transform over time
        for n1 in range(len(psi1)):
            # Retrieve first-order coefficient in the list
            j1 = psi1[n1]['j']
            if j1 >= j2:
                continue
            U_1_hat = U_1_hat_list[n1]

            # what we subsampled in 1st-order
            sub1_adj = min(j1, log2_T) if average else j1
            k1 = max(sub1_adj - oversampling, 0)
            # what we subsample now in 2nd
            sub2_adj = min(j2, log2_T) if average else j2
            k2 = max(sub2_adj - k1 - oversampling, 0)

            # Convolution and downsampling
            Y_2_c = B.cdgmm(U_1_hat, psi2[n2][k1])
            Y_2_hat = B.subsample_fourier(Y_2_c, 2**k2)
            Y_2_c = B.ifft(Y_2_hat)
            Y_2_list.append(Y_2_c)
            # Y_2_arr[n1] = Y_2_c

        Y_2_arr = _right_pad(Y_2_list, pad_fr, B)

        # sum is same for all `n1`, just take last
        k1_plus_k2 = k1 + k2

        # swap axes & map to Fourier domain to prepare for conv along freq
        Y_2_hat = _transpose_fft(Y_2_arr, B, B.fft)

        # Transform over frequency + low-pass, for both spins
        # `* psi_f` part of `X * (psi_t * psi_f)`
        _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons,
                              out_S_2['psi_t * psi_f'])

        # Low-pass over frequency
        # `* phi_f` part of `X * (psi_t * phi_f)`
        _frequency_lowpass(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons,
                           out_S_2['psi_t * phi_f'])

    ##########################################################################
    # `U1 * (phi_t * psi_f)`
    # take largest subsampling factor
    j2 = log2_T

    # preallocate output slice
    pad_fr = sc_freq.J_pad_fr_max
    n2_time = U_0.shape[-1] // 2**max(j2 - oversampling, 0)
    Y_2_list = []
    # Y_2_arr = B.zeros_like(U_1_c, (2**pad_fr, n2_time))

    # Low-pass filtering over time, with filter length matching first-order's
    for n1 in range(len(psi1)):
        j1 = psi1[n1]['j']
        # Convolution and downsampling
        # what we subsampled in 1st-order
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)
        # what we subsample now in 2nd
        k2 = max(j2 - k1 - oversampling, 0)
        # reuse 1st-order U_1_hat * phi[k1]
        Y_2_c = S_1_c_list[n1]
        Y_2_hat = B.subsample_fourier(Y_2_c, 2**k2)
        Y_2_c = B.ifft(Y_2_hat)
        Y_2_list.append(Y_2_c)
        # Y_2_arr[n1] = Y_2_c

    # sum is same for all `n1`, just take last
    k1_plus_k2 = k1 + k2

    Y_2_arr = _right_pad(Y_2_list, pad_fr, B)
    # swap axes & map to Fourier domain to prepare for conv along freq
    Y_2_hat = _transpose_fft(Y_2_arr, B, B.fft)

    # Transform over frequency + low-pass
    # `* psi_f` part of `X * (phi_t * psi_f)`
    _frequency_scattering(Y_2_hat, j2, -1, pad_fr, k1_plus_k2, commons,
                          out_S_2['phi_t * psi_f'], spin_down=False)

    ##########################################################################
    # pack outputs & return
    out = {}
    out['S0'] = out_S_0
    out['S1'] = out_S_1
    out['phi_t * phi_f'] = out_S_2['phi_t * phi_f']
    out['phi_t * psi_f'] = out_S_2['phi_t * psi_f'][0]
    out['psi_t * phi_f'] = out_S_2['psi_t * phi_f']
    out['psi_t * psi_f_up']   = out_S_2['psi_t * psi_f'][0]
    out['psi_t * psi_f_down'] = out_S_2['psi_t * psi_f'][1]

    if out_type == 'dict:array':
        for k, v in out.items():
            if out_3D:
                # stack joint slices, preserve 3D structure
                out[k] = B.concatenate([c['coef'] for c in v], axis=1)
            else:
                # flatten joint slices, return 2D
                out[k] = B.concatenate_v2([c['coef'] for c in v], axis=1)
    elif out_type == 'dict:list':
        pass  # already done
    elif out_type == 'array':
        if out_3D:
            # cannot concatenate `S0` & `S1` with joint slices, return two arrays
            out_0 = B.concatenate_v2([c['coef'] for k, v in out.items()
                                      for c in v if k in ('S0', 'S1')], axis=1)
            out_1 = B.concatenate([c['coef'] for k, v in out.items()
                                   for c in v if k not in ('S0', 'S1')], axis=1)
            out = (out_0, out_1)
        else:
            # flatten all and concat along `freq` dim
            out = B.concatenate_v2([c['coef'] for v in out.values()
                                    for c in v], axis=1)
    elif out_type == 'list':
        out = [c for v in out.values() for c in v]

    return out


def _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons, out_S_2,
                          spin_down=True):
    B, sc_freq, aligned, oversampling_fr, average_fr, *_ = commons

    psi1_fs = [sc_freq.psi1_f_fr_up]
    if spin_down:
        psi1_fs.append(sc_freq.psi1_f_fr_down)

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, psi1_f in enumerate(psi1_fs):
        for n1_fr in range(len(psi1_f)):
            subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr

            # determine subsampling reference
            if aligned:
                # subsample as we would in min-padded case
                reference_subsample_equiv_due_to_pad = max(
                    sc_freq.subsampling_equiv_relative_to_max_padding)
            else:
                # subsample regularly (relative to current padding)
                reference_subsample_equiv_due_to_pad = subsample_equiv_due_to_pad

            # compute subsampling and fetch filter
            j1_fr = psi1_f[n1_fr]['j'][subsample_equiv_due_to_pad]
            if average_fr:
                sub_adj = min(j1_fr, sc_freq.max_subsampling_before_phi_fr)
            else:
                sub_adj = j1_fr if not aligned else 0
            n1_fr_subsample = max(sub_adj - reference_subsample_equiv_due_to_pad -
                                  oversampling_fr, 0)

            # Wavelet transform over frequency
            Y_fr_c = B.cdgmm(Y_2_hat, psi1_f[n1_fr][subsample_equiv_due_to_pad])
            Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample)
            Y_fr_c = B.ifft(Y_fr_hat)

            # Modulus
            U_2_m = B.modulus(Y_fr_c)

            # Convolve by Phi = phi_t * phi_f, unpad
            S_2 = _joint_lowpass(U_2_m, n2, subsample_equiv_due_to_pad,
                                 n1_fr_subsample, k1_plus_k2, commons)

            spin = (1, -1)[s1_fr] if spin_down else 0
            out_S_2[s1_fr].append({'coef': S_2,
                                   'j': (j2, j1_fr),
                                   'n': (n2, n1_fr),
                                   's': (spin,)})


def _frequency_lowpass(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons, out_S_2):
    B, sc_freq, aligned, oversampling_fr, average_fr, *_ = commons

    if aligned:
        # subsample as we would in min-padded case
        reference_subsample_equiv_due_to_pad = max(
            sc_freq.subsampling_equiv_relative_to_max_padding)
    else:
        # subsample regularly (relative to current padding)
        reference_subsample_equiv_due_to_pad = (
            sc_freq.subsampling_equiv_relative_to_max_padding[n2])

    # TODO what exactly is log2_F?
    # is it maximum permitted subsampling relative to J_pad_fr_max_init?
    # is that what it *should* be, or should it be w.r.t. nextpow2(shape_fr_max)?
    # can (and should) we end up subsampling by more than log2_F
    # (in e.g. resample_=True)?

    subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr
    # take largest permissible subsampling factor given input length
    # (always log2_F if `resample_phi_fr=True`,
    # else less by `subsample_equiv_due_to_pad`)
    j1_fr = sc_freq.phi_f_fr['j'][subsample_equiv_due_to_pad]
    sub_adj = (j1_fr if not average_fr else
               min(j1_fr, sc_freq.max_subsampling_before_phi_fr))
    n1_fr_subsample = max(sub_adj - reference_subsample_equiv_due_to_pad -
                          oversampling_fr, 0)

    Y_fr_c = B.cdgmm(Y_2_hat, sc_freq.phi_f_fr[subsample_equiv_due_to_pad])
    Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample)
    Y_fr_c = B.ifft(Y_fr_hat)

    # Modulus
    U_2_m = B.modulus(Y_fr_c)

    # Convolve by Phi = phi_t * phi_f
    S_2 = _joint_lowpass(U_2_m, n2, subsample_equiv_due_to_pad, n1_fr_subsample,
                         k1_plus_k2, commons)

    out_S_2.append({'coef': S_2,
                    'j': (j2, j1_fr),
                    'n': (n2, -1),
                    's': (0,)})


def _joint_lowpass(U_2_m, n2, subsample_equiv_due_to_pad, n1_fr_subsample,
                   k1_plus_k2, commons):
    def unpad_fr(S_2_fr, total_subsample_fr):
        if out_3D:
            return unpad(S_2_fr,
                         sc_freq.ind_start_fr_max[total_subsample_fr],
                         sc_freq.ind_end_fr_max[total_subsample_fr])
        else:
            return unpad(S_2_fr,
                         sc_freq.ind_start_fr[n2][total_subsample_fr],
                         sc_freq.ind_end_fr[n2][total_subsample_fr])

    (B, sc_freq, aligned, oversampling_fr, average_fr, oversampling, average,
     unpad, log2_T, phi, ind_start, ind_end, out_3D) = commons

    # compute subsampling logic ##############################################
    total_subsample_so_far = subsample_equiv_due_to_pad + n1_fr_subsample

    if sc_freq.average_fr_global:
        pass
    elif average_fr:
        if aligned:
            # subsample as we would in min-padded case
            reference_subsample_equiv_due_to_pad = max(
                sc_freq.subsampling_equiv_relative_to_max_padding)
            if out_3D:
                # this is usually 0
                subsample_equiv_due_to_pad_min = (sc_freq.J_pad_fr_max_init -
                                                  sc_freq.J_pad_fr_max)
            else:
                subsample_equiv_due_to_pad_min = (
                    reference_subsample_equiv_due_to_pad)
            reference_total_subsample_so_far = (subsample_equiv_due_to_pad_min +
                                                n1_fr_subsample)
        else:
            # subsample regularly (relative to current padding)
            reference_total_subsample_so_far = total_subsample_so_far
        total_subsample_fr_max = sc_freq.log2_F
        lowpass_subsample_fr = max(total_subsample_fr_max -
                                   reference_total_subsample_so_far -
                                   oversampling_fr, 0)
    else:
        lowpass_subsample_fr = 0
    total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr

    # fetch frequential lowpass
    if average_fr and not sc_freq.average_fr_global:
        phi_fr = sc_freq.phi_f_fr[total_subsample_so_far]

    # do lowpassing ##########################################################
    if sc_freq.average_fr_global:
        S_2_fr = B.mean(U_2_m, axis=-1)
    elif average_fr:
        # Low-pass filtering over frequency
        U_2_hat = B.rfft(U_2_m)
        S_2_fr_c = B.cdgmm(U_2_hat, phi_fr)
        S_2_fr_hat = B.subsample_fourier(S_2_fr_c, 2**lowpass_subsample_fr)
        S_2_fr = B.irfft(S_2_fr_hat)
    else:
        S_2_fr = U_2_m

    if not sc_freq.average_fr_global:
        S_2_fr = unpad_fr(S_2_fr, total_subsample_fr)
    # Swap time and frequency subscripts again
    S_2_fr = B.transpose(S_2_fr)

    if average:
        # Low-pass filtering over time
        k2_tm_J = max(log2_T - k1_plus_k2 - oversampling, 0)
        U_2_hat = B.rfft(S_2_fr)
        S_2_c = B.cdgmm(U_2_hat, phi[k1_plus_k2])
        S_2_hat = B.subsample_fourier(S_2_c, 2**k2_tm_J)
        S_2_r = B.irfft(S_2_hat)
        total_subsample_tm = k1_plus_k2 + k2_tm_J
    else:
        total_subsample_tm = k1_plus_k2
        S_2_r = S_2_fr

    S_2 = unpad(S_2_r, ind_start[total_subsample_tm],
                ind_end[total_subsample_tm])

    # sanity checks (see "Subsampling, padding")
    if aligned and not sc_freq.average_fr_global:
        total_convolutional_stride_over_U1 = (n1_fr_subsample +
                                              lowpass_subsample_fr)
        assert (total_convolutional_stride_over_U1 ==
                sc_freq.__total_convolutional_stride_over_U1)
        if not average_fr:
            assert total_convolutional_stride_over_U1 == 0
        elif out_3D:
            max_init_diff = sc_freq.J_pad_fr_max_init - sc_freq.J_pad_fr_max
            assert (total_convolutional_stride_over_U1 ==
                    sc_freq.log2_F - max_init_diff)
    return S_2


def _transpose_fft(coeff_arr, B, fft):
    # swap dims to convolve along frequency
    out = B.transpose(coeff_arr)
    # Map to Fourier domain
    out = fft(out)
    return out


def _right_pad(coeff_list, pad_fr, B):
    zero_row = B.zeros_like(coeff_list[0])
    zero_rows = [zero_row] * (2**pad_fr - len(coeff_list))
    return B.concatenate_v2(coeff_list + zero_rows, axis=1)


__all__ = ['timefrequency_scattering']
