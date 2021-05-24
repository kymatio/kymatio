
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
        Refers to `total_conv_stride_over_U1`.

    total_conv_stride_over_U1
        See above. Alt name: `total_conv_stride_fr`; `over_U1` seeks
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

    # TODO is `aligned=True` compatible with `resample_=False`?

    aligned=True:  # TODO do these hold for `* phi_f` pairs?
        Imposes:
          - `total_conv_stride_over_U1` to be same for all joint
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
            - Additionally imposes that `total_conv_stride_over_U1==0`,
              since `psi_fr[0]['j'][0] == 0` and `_joint_lowpass()` can no longer
              compensate for variable `j_fr`.
            - Enforces separate `total_conv_stride_over_U1` for frequential
              lowpass coeffs: `phi_t * phi_f`, `psi_t * phi_f` will have their
              own common stride. This avoids significant redundancy while still
              complementing joint coeffs with lowpass information.

        out_3D=True:
            Additionally imposes that all frequential padding is the same
            (maximum), since otherwise same `total_conv_stride_over_U1`
            yields variable `freq` across `n2`.

        average_fr_global=True:
            Stride logic in `_joint_lowpass()` is relaxed since all U1 are
            collapsed into one point; will no longer check
            `total_conv_stride_over_U1`  # TODO makes sense?

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

    Debug tips
    ----------
      - Check following sc_freq attributes:
          - shape_fr
          - J_pad_fr, J_pad_fr_max_init
          - ind_end_fr, ind_end_fr_max
    """
    # pack for later
    B = backend
    average_fr = sc_freq.average_fr
    commons = (B, sc_freq, aligned, oversampling_fr, average_fr, out_3D,
               oversampling, average, unpad, log2_T, phi, ind_start, ind_end)

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
        stride = sc_freq.log2_F
    else:
        S_1_tm_T_hat = _transpose_fft(S_1_fr, B, B.rfft)

        # this is usually 0
        subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr
        n1_fr_subsample = 0  # no intermediate scattering
        total_subsample_so_far = subsample_equiv_due_to_pad + n1_fr_subsample

        # subsample as in min-padded case, or regularly, based on `aligned`
        reference_subsample_equiv_due_to_pad = _get_ref_sub(
            subsample_equiv_due_to_pad, aligned, out_3D, sc_freq)
        reference_total_subsample_so_far = (reference_subsample_equiv_due_to_pad +
                                            n1_fr_subsample)

        total_subsample_fr_max = sc_freq.log2_F
        lowpass_subsample_fr = max(total_subsample_fr_max -
                                   reference_total_subsample_so_far -
                                   oversampling_fr, 0)
        total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr
        stride = n1_fr_subsample + lowpass_subsample_fr

        # Low-pass filtering over frequency
        phi_fr = sc_freq.phi_f_fr[subsample_equiv_due_to_pad]
        S_1_fr_T_c = B.cdgmm(S_1_tm_T_hat, phi_fr)
        S_1_fr_T_hat = B.subsample_fourier(S_1_fr_T_c, 2**lowpass_subsample_fr)
        S_1_fr_T = B.irfft(S_1_fr_T_hat)

        # unpad + transpose, append to out
        if out_3D:
            ind_start_fr = sc_freq.ind_start_fr_max[total_subsample_fr]
            ind_end_fr   = sc_freq.ind_end_fr_max[  total_subsample_fr]
        else:
            ind_start_fr = sc_freq.ind_start_fr[-1][total_subsample_fr]
            ind_end_fr   = sc_freq.ind_end_fr[-1][  total_subsample_fr]
        S_1_fr_T = unpad(S_1_fr_T, ind_start_fr, ind_end_fr)
        S_1_fr = B.transpose(S_1_fr_T)
    j1_fr = (sc_freq.log2_F if sc_freq.average_fr_global else
             sc_freq.phi_f_fr['j'][subsample_equiv_due_to_pad])
    out_S_2['phi_t * phi_f'].append({'coef': S_1_fr,
                                     'j': (log2_T, j1_fr),
                                     'n': (-1, -1),
                                     's': (0,)})
    # set reference for later
    sc_freq.__total_conv_stride_over_U1_phi = stride
    sc_freq.__total_conv_stride_over_U1 = stride if average_fr else -1

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
    B, sc_freq, aligned, oversampling_fr, average_fr, out_3D, *_ = commons

    psi1_fs = [sc_freq.psi1_f_fr_up]
    if spin_down:
        psi1_fs.append(sc_freq.psi1_f_fr_down)

    # TODO what happens when `not all(J_pad_fr == J_pad_fr_max)` and
    # `resample_=True`?

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, psi1_f in enumerate(psi1_fs):
        for n1_fr in range(len(psi1_f)):
            subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr

            # subsample as in min-padded case, or regularly, based on `aligned`
            reference_subsample_equiv_due_to_pad = _get_ref_sub(
                subsample_equiv_due_to_pad, aligned, out_3D, sc_freq)

            # TODO shouldn't it be more like
            j1_fr = ...
            j1_fr_for_subsampling = psi1_f[n1_fr]['j']

            # aren't we double-accounting via j1_fr = fn(subsample_equiv_...)
            # then again `- subsample_equiv_...`

            # compute subsampling
            j1_fr = psi1_f[n1_fr]['j'][subsample_equiv_due_to_pad]
            if average_fr:
                # TODO shouldn't the min operate on n1_fr_subsample
                sub_adj = min(j1_fr, sc_freq.max_subsampling_before_phi_fr)
            else:
                sub_adj = j1_fr if not aligned else 0
            n1_fr_subsample = max(sub_adj - reference_subsample_equiv_due_to_pad -
                                  oversampling_fr, 0)

            """
              - `j1_fr = psi1_f[subs...]` is how much we can subsample at most
                at this stage due to `psi`, disregarding `phi`, but *not*
                disregarding `J_pad_fr`
              - to account for `phi` we ensure `n1_fr_subsample` does not exceed
                `max_subsampling_before_phi_fr`
              - to account for `aligned` we ensure `n1_fr_subsample` for each
                `n1_fr` is the same for all `J_pad_fr`
                (`subsample_equiv_due_to_pad`)
            """
            # "subsample as we would in min-padded case" ->
            # "TOTAL subsample (conv stride) as we would in min-padded case"

            # Wavelet transform over frequency
            Y_fr_c = B.cdgmm(Y_2_hat, psi1_f[n1_fr][subsample_equiv_due_to_pad])
            Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample)
            Y_fr_c = B.ifft(Y_fr_hat)

            # Modulus
            U_2_m = B.modulus(Y_fr_c)

            # Convolve by Phi = phi_t * phi_f, unpad
            S_2 = _joint_lowpass(U_2_m, n2, n1_fr, subsample_equiv_due_to_pad,
                                 n1_fr_subsample, k1_plus_k2, commons)

            spin = (1, -1)[s1_fr] if spin_down else 0
            out_S_2[s1_fr].append({'coef': S_2,
                                   'j': (j2, j1_fr),
                                   'n': (n2, n1_fr),
                                   's': (spin,)})


def _frequency_lowpass(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons, out_S_2):
    B, sc_freq, aligned, oversampling_fr, average_fr, out_3D, *_ = commons

    # TODO average_fr_global?

    # TODO what exactly is log2_F?
    # is it maximum permitted subsampling relative to J_pad_fr_max_init?
    # is that what it *should* be, or should it be w.r.t. nextpow2(shape_fr_max)?
    # can (and should) we end up subsampling by more than log2_F
    # (in e.g. resample_=True)? # [1]

    # [1] not a worry for `out_3D and aligned`, we pad to common max

    subsample_equiv_due_to_pad = sc_freq.J_pad_fr_max_init - pad_fr

    if sc_freq.average_fr_global:
        Y_fr_c = B.mean(Y_2_hat, axis=-1)  # TODO axis may change
        j1_fr = sc_freq.log2_F
        # TODO what about oversampling_fr? document that it's non-effective
        n1_fr_subsample = j1_fr - subsample_equiv_due_to_pad
    else:
        # subsample as in min-padded case, or regularly, based on `aligned`
        reference_subsample_equiv_due_to_pad = _get_ref_sub(
            subsample_equiv_due_to_pad, aligned, out_3D, sc_freq)

        # take largest permissible subsampling factor given input length
        # (always log2_F if `resample_phi_fr=True`,
        # else less by `subsample_equiv_due_to_pad`)
        j1_fr = sc_freq.phi_f_fr['j'][subsample_equiv_due_to_pad]

        # j1_fr limits max subsampling after convolving with phi,
        # so we still compute with max possible j1_fr but account for this
        # this adjustment is needed because we subtract from `sub_adj`
        j1_fr_n1_fr_max = max(sc_freq.phi_f_fr['j'])
        sub_adj = (j1_fr_n1_fr_max if not average_fr else
                   min(j1_fr_n1_fr_max, sc_freq.max_subsampling_before_phi_fr))
        n1_fr_subsample = max(sub_adj - reference_subsample_equiv_due_to_pad -
                              oversampling_fr, 0)
        # do not allow exceeding actual filter's subsampling factor
        n1_fr_subsample = min(n1_fr_subsample, j1_fr)

        Y_fr_c = B.cdgmm(Y_2_hat, sc_freq.phi_f_fr[subsample_equiv_due_to_pad])
        Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample)
        Y_fr_c = B.ifft(Y_fr_hat)

    # Modulus
    U_2_m = B.modulus(Y_fr_c)

    # Convolve by Phi = phi_t * phi_f
    S_2 = _joint_lowpass(U_2_m, n2, -1, subsample_equiv_due_to_pad,
                         n1_fr_subsample, k1_plus_k2, commons)

    out_S_2.append({'coef': S_2,
                    'j': (j2, j1_fr),
                    'n': (n2, -1),
                    's': (0,)})


def _joint_lowpass(U_2_m, n2, n1_fr, subsample_equiv_due_to_pad, n1_fr_subsample,
                   k1_plus_k2, commons):
    (B, sc_freq, aligned, oversampling_fr, average_fr, out_3D, oversampling,
     average, unpad, log2_T, phi, ind_start, ind_end) = commons

    # compute subsampling logic ##############################################
    total_subsample_so_far = subsample_equiv_due_to_pad + n1_fr_subsample

    """
    In regular scattering, log2_T controls maximum total subsampling upon
    padded input; if padding beyond nextpow2, will unpad more, not subsample more

    Any reason for `total_conv_stride_over_U1` to not equal `log2_F`?
      - in `aligned=True` && `out_3D=False`, we pad variably, so min padded
        cannot stride same as max padded else we may collapse `freq` to < 0.
        Then what decides maximum stride?

        Note, if `log2_F` is 1 less than global averaging we get `freq=2`
        for max padded case, then this stride on min padded case would yield < 0.
        Thus we restrict maximum stride based on min padded case, and total
        strides (psi + phi) for all `n1_fr` must equal
        `log2_F - subsample_equiv_due_to_pad_min`.

      - in `aligned=True` && `out_3D=True` we always pad to maximum, thus
        we can (and should) have `total_conv_stride_over_U1 == log2_F`.

    Since we must have same `total_conv_stride_over_U1` for all `n1_fr`,
    in variably padded case we'll necessarily get variable `freq` per `n1_fr`,
    forbidding 3D concatenation (and which is why `out_3D=True` always pads same).

    In `aligned=False` we don't care about enforcing the same stride, so we
    subsample maximally for each padding - that is, as much as the filters will
    permit. The only restriction is, still, to not end up with <= 0 `freq`.
      - "Pad more -> unpad mode" still holds in fr scattering, so
        `total_subsample_fr_max` is set relative to `nextpow2(shape_fr_max)`.
        That is, shorter `freq` due to the combination of padding less, and
        conv stride (subsampling), is the same for all paddings, and for greater
        `J_pad_fr` we simply unpad more.

    What is `log2_F` set relative to, and can `total_subsample_fr_max == log2_F`?
      - `log2_F` is set by the user indirectly via `F` to control amount of
        imposed frequency transposition invariance, via temporal support of
        lowpass.
      - `log2_F` controls max subsampling we can do after conv with `phi_fr`.

        What does this mean for inputs (to `phi_fr`) of varying length - namely,
        length lesser than maximum (which is the length `phi_fr` was originally
        sampled at, and length at which it could be subsampled by `2**log2_F`)?
          - With `resample_=True`, we can still subsample by `log2_F` without
            alias - restrictions will come from elsewhere
            (`aligned`, `freq` <= 0, etc).
          - With `resample_=False`, allowed subsampling is less than `log2_F`,
            and this is accounted for alongside existing constraints.

        Thus: account for `phi_fr`'s `j` at any length along other constraints.
      - `log2_F` cannot exceed `nextpow2(shape_fr_max)`; can
        `total_subsample_fr_max` exceed it? For max padded case, whose pad length
        might exceed `nextpow2(shape_fr_max)`, we can still total subsample by
        `log2_F` at most (`average_fr=True`; note this quantity is unused for
        `False`); the rest will "unpad more".

        Thus, `total_subsample_fr_max <= log2_F`. Can it be <? The *actual*
        subsampling we do can be, but it'll be due to factors unrelated to
        subsampling permitted by `phi_fr`.

        Thus, `total_subsample_fr_max == log2_F`. (effectively by definition)

      - However we still need a quantity that denotes *actual* maximum subsampling
        we'll do at any pad length, given configuration (`aligned` etc).
        Call it `total_subsample_fr_max_adj`.
          - Differs from `total_subsample_fr_max` when `aligned`: total conv
            stride is restricted by min padded case.
    """
    """
    Exhaustive diagrams
    ===================

    _joint_lowpass()
    ----------------

    **Conditions**:

    `phi_fr[subsample_equiv_due_to_pad][n1_fr_subsample]` indexing for all cases.
    What differs is subsampling afterwards, determined by:
        C1) `j` of the `phi_fr`
            - depends on `resample_phi_fr`
        C2) whether `freq` should be same for all `n2, n1_fr`
            - depends on `out_3D`
        C3) whether `total_conv_stride_over_U1` should be same for all `n2, n1_fr`
            - depends on `aligned`
        C4) whether `freq` <= 0
            - independent of `aligned, out_3D, resample_*`
            - must enforce total subsampling (equivalent and due to stride)
              does not exceed `pad_fr`
            - indirect dependence on `resample_phi_fr` where, in `True`,
              we can subsample by more than `log2_F`, which is accounted for
              by `max_subsampling_before_phi_fr`

    **Definitions**:
        D1) `total_subsample_fr` refers to total subsampling (equivalent due to
            padding less, and due to convolutional stride) relative to
            `J_pad_fr_max_init`. That is,
            `total_subsample_fr == freq / J_pad_fr_max_init`, where `freq` is
            length of frequential dimension *before* unpadding. It is also ==
            `lowpass_subsample_fr + n1_fr_subsample + subsample_equiv_due_to_pad`.
        D2) `log2_N_fr == nextpow2(shape_fr_max)` is the maximum possible
            `total_conv_stride_over_U1` for any configuration.
            (Exceeding would imply `freq` <0 0 over unpadded `shape_fr_max`).
              - `log2_N_fr + diff == J_pad_fr_max_init` is the maximum possible
                `total_subsample_fr` for any configuration.
              - `log2_F <= log2_N_fr` for all configurations.

    ############################################################################
    X. aligned==True:
      __________________________________________________________________________
      A. out_3D==True:
        resample_psi_fr, resample_phi_fr:
        1. True, True:
           `lowpass_subsample_fr == log2_F - n1_fr_subsample`. Because:
            a. `lowpass_subsample_fr` cannot exceed `log2_F` because `log2_F`
                is the maximum subsampling factor of any `phi_fr`.
            b. `lowpass_subsample_fr` can be less than `log2_F` because must
               subsample less to get same `freq` with different `n1_fr_subsample`
            c. `lowpass_subsample_fr` and `n1_fr_subsample` must sum to the same
               (`total_conv_stride_over_U1`)
            d. `total_conv_stride_over_U1` cannot exceed `log2_F` because minimum
               `n1_fr_subsample == 0`, and max `lowpass_subsample_fr == log2_F`.
            e. `phi_fr` will be subsample-able by `log2_F` for any `pad_fr`
               (`J_pad_fr` will be restricted by `max_subsampling_before_phi_fr`
                to ensure this).
            f. `freq` <= 0 isn't possible because we always have
               `pad_fr == J_pad_fr_max == J_pad_fr_max_init` (or equivalently
               `pad_fr == J_pad_fr_max <= J_pad_fr_max_init` if not `True, True`,
               where `J_pad_fr_max >= log2_N_fr`, and `log2_F <= log2_N_fr` (D2)).

           `total_subsample_fr == log2_F`.
            g. `total_subsample_fr` cannot exceed `log2_F` due to
               `pad_fr == J_pad_fr_max == J_pad_fr_max_init`.

           `total_conv_stride_over_U1 == log2_F`
            d+. `lowpass_subsample_fr == log2_F` will happen along
                `n1_fr_subsample == 0`. Per c, `total_conv_stride_over_U1` hence
                also cannot be less, and must `== log2_F`.

        2. False, True:
           Same as 1. Because:
            a. Same as 1's. `J_pad_fr_max < J_pad_fr_max_init` is possible,
               but 1e applies.
            b,c,d,e: same as 1's.
            f. Same as 1's (`pad_fr == J_pad_fr_max >= log2_N_fr`).

           `total_subsample_fr == log2_F + diff`, where
           `diff == J_pad_fr_max_init - J_pad_fr_max`.
            g. `total_subsample_fr` can exceed `log2_F` because we can have
               `pad_fr == J_pad_fr_max < J_pad_fr_max_init`, and `_, True`
               will keep the `< 2**J_pad_fr_max_init` length `phi_fr`'s
               subsampling factor at `log2_F` (i.e. 1e).

           `total_conv_stride_over_U1 == log2_F`
            d+. Same as 1's (also see 2f).

        3. True, False:
           `lowpass_subsample_fr == (log2_F - diff) - n1_fr_subsample`. Because:
            a,b,c: same as 1's.
            d. same as 1's except `log2_F` -> `log2_F - diff` (see 3e).
            e. `phi_fr` will not be subsample-able by `log2_F` for all `pad_fr`s.
               `phi_fr` with `j=log2_F` is sampled at `2**J_pad_fr_max_init`,
               and subsequently has `j<log2_F` (see 3f).
               That is, `_, False` makes `phi_fr`'s `j` directly depend on input
               length, so both `n1_fr_subsample` and `subsample_equiv_due_to_pad`
               lower `j`. `_, False` (or `False, _`) enables
               `J_pad_fr_max < J_pad_fr_max_init`.
            f. Same as 1's (+2's reasoning).

            `total_subsample_fr == log2_F`. Because,
             g. Per 3e, exceeding `log2_F` would mean subsampling `phi_fr`
                beyond `log2_F`.

            `total_conv_stride_over_U1 == log2_F - diff`
             d+. Same as 1's, except `log2_F` -> `log2_F - diff` per 3e.

        4. False, False:
           Same as 3.
            a,b,c,d,e,f: same as 3's. `False, _` only affects `J_pad_fr`
            (but `pad_fr == J_pad_fr_max`, always) and `J_pad_fr_max`
            (and 3 accounts for this).

        ```
        lowpass_subsample_fr == phi_fr_sub_max - n1_fr_subsample
        phi_fr_sub_max == phi_fr['j'][J_pad_fr_max] <= log2_F
        # ^ maximum "realizable" subsampling after `phi_fr`
        ```
        accounts for 1-4.
          - `phi_fr_sub_max == log2_F` for 1,2 and we reduce to
            `lowpass_subsample_fr == log2_F - n1_fr_subsample`
          - `phi_fr_sub_max == log2_F - diff` for 3,4.

        ```
        total_subsample_fr == log2_F + (diff - j_diff)
        j_diff == phi_fr['j'][J_pad_fr_max_init] - phi_fr['j'][J_pad_fr_max]
        ```
        accounts for 1-4.
          - 1: `J_pad_fr_max == J_pad_fr_max_init` -> `diff == j_diff == 0`
               -> `total_subsample_fr == log2_F`
          - 2: `J_pad_fr_max <= J_pad_fr_max_init` -> `diff >= 0`, `j_diff == 0`
               -> `total_subsample_fr == log2_F + diff
          - 3: `J_pad_fr_max <= J_pad_fr_max_init` -> `diff == j_diff >= 0`
               -> `total_subsample_fr == log2_F`
          - 4: same as 3.

        ```
        total_conv_stride_over_U1 == log2_F - j_diff
        ```
        accounts for 1-4.
          - `== log2_F` for 1,2, and `== log2_F - diff` for 3,4.

        A "silent" condition has been obeyed via `J_pad_fr == J_pad_fr_max`:
        Y.1.e. # TODO
      __________________________________________________________________________
      B. out_3D==False:
         ^1: `pad_fr == J_pad_fr_max` no longer always holds
         ^2: same `freq` for all `n1_fr, n2` no longer required

        resample_psi_fr, resample_phi_fr:
        1. True, True
           `lowpass_subsample_fr == min(log2_F - n1_fr_subsample, pad_fr)`.
           `lowpass_subsample_fr == max(0, min(log2_F, J_pad_fr_min) -
                                           n1_fr_subsample)`
           Because:
            a. == A1a.
            b. `lowpass_subsample_fr` does not need to be less than `log2_F`
               due to B^2.
            c. == A1c. This overrides b.
            d. == A1d.
            e. == A1e.
            f. `freq` <= 0 is possible without `min` per B^1. `freq` <= 0 will
               occur upon `lowpass_subsample_fr > pad_fr`, so max `== pad_fr`.

           `total_subsample_fr == log2_F + subsample_equiv_due_to_pad`
            g. `total_subsample_fr` can exceed `log2_F` because we can have
               `pad_fr < J_pad_fr_max == J_pad_fr_max_init` (or equivalently
               `pad_fr < J_pad_fr_max <= J_pad_fr_max_init` for not `True, True`).
               (`subsample_equiv_due_to_pad == J_pad_fr_max_init - pad_fr` is
               the generalization of `diff` (see e.g. A2g))

           `total_conv_stride_over_U1 == min(log2_F, J_pad_fr_min)`.
            d+. Smallest conv stride is determined from `J_pad_fr_min` such that
                `total_subsample_fr <= J_pad_fr_min`, i.e. sum of
                `lowpass_subsample_fr`, `n1_fr_subsample`, and
                `subsample_equiv_due_to_pad` is `<= J_pad_fr_min`.
                Smallest conv stride will occur at `n1_fr_subsample==0`, equal to
                `lowpass_subsample_fr`'s maximum at `J_pad_fr_min` (which is
                `log2_F` if `log2_F <= J_pad_fr_min`, else `J_pad_fr_min`).
                c then forbids exceeding this (and can't lower since this is min).

        2. False, True:
           Same as 1.
            a. Same as 1's. `J_pad_fr_max < J_pad_fr_max_init` is possible,
               but 1e applies.
            b,c,d,e,f,g: same as 1's.

        3. True, False:
           `lowpass_subsample_fr == ((log2_F - subsample_equiv_due_to_pad) -
                                     n1_fr_subsample)`. Because:
            a,b,c,d,f: same as 1's.
            e: same as 1's. `subsample_equiv_due_to_pad` is the generalization
               of `diff` for `pad_fr < J_pad_fr_max`, i.e. not just
               `pad_fr == J_pad_fr_max < J_pad_fr_max_init`.

            `total_subsample_fr

        4. False, False:
            Same as 3.
             a,b,c,d: same as 3's. `False, _` only affects `pad_fr` via
             `J_pad_fr`.

        ```
        lowpass_subsample_fr == (min(log2_F + diff, phi_fr_sub_max) -
                                 n1_fr_subsample)
        ```
        accounts for 1-4.

    Thus:
    ```
    lowpass_subsample_fr == (min(log2_F + diff, phi_fr_sub_max) -
                             n1_fr_subsample)
    ```
    accounts for A1-4, and B1-4.

    ############################################################################
    Y. aligned==False:

    This relaxes c,d in all of X:
        c. `lowpass_subsample_fr` and `n1_fr_subsample` no longer need to sum
           to the same value (`total_conv_stride_over_U1`)
        d. `total_conv_stride_over_U1` can exceed `phi_fr_sub_max` per c.

      __________________________________________________________________________
      A. out_3D==True:
        resample_psi_fr, resample_phi_fr:
        1. True, True:
           `lowpass_subsample_fr == log2_F - n1_fr_subsample`. Because:
            a,b: same as X.A.1's.
            e. `lowpass_subsample_fr`, `n1_fr_subsample`, and
               `subsample_equiv_due_to_pad` must sum to the same value
               (`total_subsample_fr`) for all `n2, n1_fr` (to get same `freq`).
               Since `subsample_equiv_due_to_pad` is same for all `n2, n1_fr`,
               only source of variability is `n1_fr_subsample`.

        2. False, True:
           Same as X.A.2.
            a,b: same as X.A.2's.
            e: same as 1's.

        3. True, False:
           Same as 1.
            a,b,e: same as 1's, because `_, False` limits
            `lowpass_subsample_fr` to `log2_F`, regardless of `pad_fr`.

        4. False, False:
           Same as 1 (+3's reasoning).
            a,b,e: same as 1's. `False, _` only affects `J_pad_fr`.

        ```
        lowpass_subsample_fr == (min(log2_F + diff, phi_fr_sub_max) -
                                 n1_fr_subsample)
        ```
        accounts for 1-4.

      __________________________________________________________________________
      B. out_3D==False:
        resample_psi_fr, resample_phi_fr:
        1. True, True
           `lowpass_subsample_fr == log2_F + diff`. Because:
            a. `lowpass_subsample_fr` can exceed `log2_F` because we can have
               `pad_fr < J_pad_fr_max`.
            b. `lowpass_subsample_fr` does not need to be less than
               `log2_F + diff`, because we no longer care for same `freq`.
            e. `lowpass_subsample_fr`, `n1_fr_subsample`, and
               `subsample_equiv_due_to_pad` no longer need to sum to the same
               value (`total_subsample_fr`) for all `n2, n1_fr`, since we don't
               care to get the same `freq`.

        2. False, True:
           Same as 1.
            a,b: same as 1's. `False, _` only affects `J_pad_fr` (and
            `J_pad_fr_max`).

        3. True, False:
           `lowpass_subsample_fr == log2_F`. Because:
            a. `lowpass_subsample_fr` cannot exceed `log2_F` due to `_, False`.
            b,e. same as 1's, except `log2_F + diff -> log2_F`.

        4. False, False:
            Same as 3.
             a,b,e: same as 3's. `False, _` only affects `pad_fr` via  `J_pad_fr`.

        ```
        lowpass_subsample_fr == (min(log2_F + diff, phi_fr_sub_max) -
                                 n1_fr_subsample)
        ```
        accounts for 1-4.
    """
    # TODO psi_fr's subsampling must additionally be constrained by log2_F
    if sc_freq.average_fr_global:
        pass
    elif average_fr:
        # subsample as in min-padded case, or regularly, based on `aligned`
        reference_subsample_equiv_due_to_pad = _get_ref_sub(
            subsample_equiv_due_to_pad, aligned, out_3D, sc_freq)
        reference_total_subsample_so_far = (reference_subsample_equiv_due_to_pad +
                                            n1_fr_subsample)
        # TODO "total subsample" relative to what? it's not the conv stride
        total_subsample_fr_max = sc_freq.log2_F
        # accounts for `aligned`; total amount of subsampling cannot exceed this,
        # and for `aligned=True`,
        total_subsample_fr_max_adj = (total_subsample_fr_max -
                                      reference_subsample_equiv_due_to_pad)
        # TODO we don't even query `phi_f['j']`...
        lowpass_subsample_fr = max(total_subsample_fr_max -
                                   reference_total_subsample_so_far -
                                   oversampling_fr, 0)
        # TODO `total_subsample_fr_max` -> `total_subsample_fr`?

        # TODO shouldn't we subsample `phi_fr` for subsampled `U_2_m`??
    else:
        lowpass_subsample_fr = 0
    total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr

    # fetch frequential lowpass
    if average_fr and not sc_freq.average_fr_global:
        # phi_fr = sc_freq.phi_f_fr[total_subsample_so_far]
        phi_fr = sc_freq.phi_f_fr[subsample_equiv_due_to_pad][n1_fr_subsample]

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
        if out_3D:
            ind_start_fr = sc_freq.ind_start_fr_max[total_subsample_fr]
            ind_end_fr   = sc_freq.ind_end_fr_max[  total_subsample_fr]
        else:
            ind_start_fr = sc_freq.ind_start_fr[n2][total_subsample_fr]
            ind_end_fr   = sc_freq.ind_end_fr[  n2][total_subsample_fr]
        S_2_fr = unpad(S_2_fr, ind_start_fr, ind_end_fr)
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
        total_conv_stride_over_U1 = n1_fr_subsample + lowpass_subsample_fr
        if n1_fr != -1:
            if sc_freq.__total_conv_stride_over_U1 == -1:
                # set if not set yet
                sc_freq.__total_conv_stride_over_U1 = total_conv_stride_over_U1
            assert (total_conv_stride_over_U1 ==
                    sc_freq.__total_conv_stride_over_U1)

            if not average_fr:
                assert total_conv_stride_over_U1 == 0
            elif out_3D:
                max_init_diff = sc_freq.J_pad_fr_max_init - sc_freq.J_pad_fr_max
                expected_common_stride = max(sc_freq.log2_F - max_init_diff -
                                             oversampling_fr, 0)
                assert (total_conv_stride_over_U1 ==
                        expected_common_stride)
        else:
            try:
                assert (total_conv_stride_over_U1 ==
                        sc_freq.__total_conv_stride_over_U1_phi)
            except:
                for k in ('J_pad_fr', 'J_pad_fr_max', 'J_pad_fr_max_init'):
                    print(getattr(sc_freq, k), '--', k)
                print(total_conv_stride_over_U1,
                      sc_freq.__total_conv_stride_over_U1_phi)
                1/0
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


def _get_ref_sub(subsample_equiv_due_to_pad, aligned, out_3D, sc_freq):            # determine subsampling reference
    if aligned:
        # subsample as we would in min-padded case
        if out_3D:
            # this is usually 0
            reference_subsample_equiv_due_to_pad = (sc_freq.J_pad_fr_max_init -
                                                    sc_freq.J_pad_fr_max)
        else:
            reference_subsample_equiv_due_to_pad = max(
                sc_freq.subsampling_equiv_relative_to_max_padding)
    else:
        # subsample regularly (relative to current padding)
        reference_subsample_equiv_due_to_pad = subsample_equiv_due_to_pad
    return reference_subsample_equiv_due_to_pad


__all__ = ['timefrequency_scattering']
