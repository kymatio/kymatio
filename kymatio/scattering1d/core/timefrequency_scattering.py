
def timefrequency_scattering(
        x, pad, unpad, backend, J, log2_T, psi1, psi2, phi, sc_freq,
        pad_left=0, pad_right=0, ind_start=None, ind_end=None,
        oversampling=0, oversampling_fr=0, aligned=True,
        average=True, average_global=None, out_type='array', pad_mode='zero'):
    """
    Main function implementing the joint time-frequency scattering transform.
    """
    # pack for later
    B = backend
    average_fr = sc_freq.average
    commons = (B, sc_freq, aligned, oversampling_fr, average_fr, oversampling,
               average, out_type, unpad, log2_T, phi, ind_start, ind_end)

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
    out_S_0.append({'coef': S_0, 'j': (), 'n': (), 's': ()})

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
    if average_fr and average:
        # zero-pad along frequency, map to Fourier domain
        pad_fr = sc_freq.J_pad_max
        S_1_fr = _right_pad(S_1_list, pad_fr, B)
        # S_1_fr = B.zeros_like(S_1_list[-1], (2**pad_fr, S_1_list[-1].shape[-1]))
        # S_1_fr[:len(S_1_list)] = S_1_list

    if sc_freq.average_global and average:
        S_1_fr = B.mean(S_1_fr, axis=-2)  # TODO axis will change
    elif average_fr and average:
        S_1_tm_T_hat = _transpose_fft(S_1_fr, B, B.rfft)

        if aligned:
            # subsample as we would in min-padded case
            reference_subsample_equiv_due_to_pad = max(sc_freq.j0s)
            if 'array' in out_type:
                subsample_equiv_due_to_pad_min = 0
            elif out_type == 'list':
                subsample_equiv_due_to_pad_min = (
                    reference_subsample_equiv_due_to_pad)
            reference_total_subsample_so_far = (subsample_equiv_due_to_pad_min +
                                                0)
        else:
            # subsample regularly (relative to current padding)
            reference_total_subsample_so_far = 0
        total_subsample_fr_max = sc_freq.log2_F
        lowpass_subsample_fr = max(total_subsample_fr_max -
                                   reference_total_subsample_so_far -
                                   oversampling_fr, 0)

        # Low-pass filtering over frequency
        S_1_fr_T_c = B.cdgmm(S_1_tm_T_hat, sc_freq.phi_f[0])
        S_1_fr_T_hat = B.subsample_fourier(S_1_fr_T_c, 2**lowpass_subsample_fr)
        S_1_fr_T = B.irfft(S_1_fr_T_hat)

        # unpad + transpose, append to out
        if out_type == 'list':
            S_1_fr_T = unpad(S_1_fr_T,
                             sc_freq.ind_start[-1][lowpass_subsample_fr],
                             sc_freq.ind_end[-1][lowpass_subsample_fr])
        elif 'array' in out_type:
            S_1_fr_T = unpad(S_1_fr_T,
                             sc_freq.ind_start_max[lowpass_subsample_fr],
                             sc_freq.ind_end_max[lowpass_subsample_fr])
        S_1_fr = B.transpose(S_1_fr_T)
    else:
        S_1_fr = []
    out_S_2['phi_t * phi_f'].append({'coef': S_1_fr, 'j': (), 'n': (), 's': ()})
    # RFC: should we put placeholders for j1 and n1 instead of empty tuples?

    ##########################################################################
    # Joint scattering: separable convolutions (along time & freq), and low-pass
    # `U1 * (psi_t * psi_f)` (up & down), and `U1 * (psi_t * phi_f)`
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        if j2 == 0:
            continue

        # preallocate output slice
        if aligned and 'array' in out_type:
            pad_fr = sc_freq.J_pad_max
        else:
            pad_fr = sc_freq.J_pad[n2]
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
    pad_fr = sc_freq.J_pad_max
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
    out['psi_t * psi_f_up']   = out_S_2['psi_t * psi_f'][0]
    out['psi_t * psi_f_down'] = out_S_2['psi_t * psi_f'][1]
    out['psi_t * phi_f'] = out_S_2['psi_t * phi_f']
    out['phi_t * psi_f'] = out_S_2['phi_t * psi_f'][0]
    out['phi_t * phi_f'] = out_S_2['phi_t * phi_f']

    if out_type == 'array':
        for k, v in out.items():
            out[k] = B.concatenate([c['coef'] for c in v], axis=1)
    return out


def _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons, out_S_2,
                          spin_down=True):
    B, sc_freq, aligned, oversampling_fr, average_fr, *_ = commons

    psi1_fs = [sc_freq.psi1_f_up]
    if spin_down:
        psi1_fs.append(sc_freq.psi1_f_down)

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, psi1_f in enumerate(psi1_fs):
        for n1_fr in range(len(psi1_f)):
            subsample_equiv_due_to_pad = sc_freq.J_pad_max - pad_fr

            # determine subsampling reference
            if aligned:
                # subsample as we would in min-padded case
                reference_subsample_equiv_due_to_pad = max(sc_freq.j0s)
            else:
                # subsample regularly (relative to current padding)
                reference_subsample_equiv_due_to_pad = subsample_equiv_due_to_pad

            # compute subsampling and fetch filter
            j1_fr = psi1_f[n1_fr]['j']
            sub_adj = (j1_fr if not average_fr else
                       min(j1_fr, sc_freq.max_subsampling_before_phi_fr))
            n1_fr_subsample = max(sub_adj - reference_subsample_equiv_due_to_pad -
                                  oversampling_fr, 0)

            # Wavelet transform over frequency
            Y_fr_c = B.cdgmm(Y_2_hat, psi1_f[n1_fr][subsample_equiv_due_to_pad])
            Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample)
            Y_fr_c = B.ifft(Y_fr_hat)

            # Modulus
            U_2_m = B.modulus(Y_fr_c)

            # Convolve by Phi = phi_t * phi_f
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
        reference_subsample_equiv_due_to_pad = max(sc_freq.j0s)
    else:
        # subsample regularly (relative to current padding)
        reference_subsample_equiv_due_to_pad = sc_freq.j0s[n2]

    subsample_equiv_due_to_pad = sc_freq.J_pad_max - pad_fr
    # take largest subsampling factor
    j1_fr = sc_freq.log2_F
    sub_adj = (j1_fr if not average_fr else
               min(j1_fr, sc_freq.max_subsampling_before_phi_fr))
    n1_fr_subsample = max(sub_adj - reference_subsample_equiv_due_to_pad -
                          oversampling_fr, 0)

    Y_fr_c = B.cdgmm(Y_2_hat, sc_freq.phi_f[subsample_equiv_due_to_pad])
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
        if out_type == 'list':
            return unpad(S_2_fr,
                         sc_freq.ind_start[n2][total_subsample_fr],
                         sc_freq.ind_end[n2][total_subsample_fr])
        elif out_type == 'array':
            return unpad(S_2_fr,
                         sc_freq.ind_start_max[total_subsample_fr],
                         sc_freq.ind_end_max[total_subsample_fr])

    (B, sc_freq, aligned, oversampling_fr, average_fr, oversampling, average,
     out_type, unpad, log2_T, phi, ind_start, ind_end) = commons

    # compute subsampling logic ##############################################
    total_subsample_fr_max = sc_freq.log2_F
    total_subsample_so_far = subsample_equiv_due_to_pad + n1_fr_subsample

    if aligned:
        # subsample as we would in min-padded case
        reference_subsample_equiv_due_to_pad = max(sc_freq.j0s)
        if out_type == 'array':
            subsample_equiv_due_to_pad_min = 0
        elif out_type == 'list':
            subsample_equiv_due_to_pad_min = reference_subsample_equiv_due_to_pad
        reference_total_subsample_so_far = (subsample_equiv_due_to_pad_min +
                                            n1_fr_subsample)
    else:
        # subsample regularly (relative to current padding)
        reference_total_subsample_so_far = total_subsample_so_far

    if sc_freq.average_global:
        pass
    elif average_fr:
        lowpass_subsample_fr = max(total_subsample_fr_max -
                                   reference_total_subsample_so_far -
                                   oversampling_fr, 0)
        total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr
    else:
        total_subsample_fr = total_subsample_so_far

    if average_fr and not sc_freq.average_global:
        # fetch frequential lowpass
        phi_fr = sc_freq.phi_f[total_subsample_so_far]

    # do lowpassing ##########################################################
    if sc_freq.average_global:
        S_2_fr = B.mean(U_2_m, axis=-1)
    elif average_fr:
        # Low-pass filtering over frequency
        U_2_hat = B.rfft(U_2_m)
        S_2_fr_c = B.cdgmm(U_2_hat, phi_fr)
        S_2_fr_hat = B.subsample_fourier(S_2_fr_c, 2**lowpass_subsample_fr)
        S_2_fr = B.irfft(S_2_fr_hat)
    else:
        S_2_fr = U_2_m

    if not sc_freq.average_global:
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
