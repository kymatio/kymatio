
def timefrequency_scattering(
        x, pad, unpad, backend, J, psi1, psi2, phi, sc_freq,
        pad_left=0, pad_right=0, ind_start=None, ind_end=None, oversampling=0,
        oversampling_fr='auto', max_order=2, average=True,
        size_scattering=(0, 0, 0), out_type='array'):
    """
    Main function implementing the joint time-frequency scattering transform.
    """
    # pack for later
    B = backend
    commons = (B, sc_freq, oversampling, oversampling_fr, out_type, unpad, J, phi,
               ind_start, ind_end, average)

    batch_size = x.shape[0]
    kJ = max(J - oversampling, 0)
    temporal_size = ind_end[kJ] - ind_start[kJ]
    out_S_1 = []
    out_S_2 = {'psi_t * psi_f': [[], []],
               'psi_t * phi_f': [],
               'phi_t * psi_f': [[]]}

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)

    # compute the Fourier transform
    U_0_hat = B.rfft(U_0)

    # First order:
    U_1_hat_list, S_1_list, S_1_c_list = [], [], []
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)
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
        if average:
            # Low-pass filtering over time
            k1_J = max(J - k1 - oversampling, 0)
            S_1_hat = B.subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = B.irfft(S_1_hat)

            # Unpad
            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
            S_1_list.append(S_1)
        else:
            # Unpad
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])
        out_S_1.append({'coef': S_1, 'j': (j1,), 'n': (n1,), 's': ()})

    # Apply low-pass filtering over frequency (optional) and unpad
    if average:
        # zero-pad along frequency, map to Fourier domain
        total_height = 2 ** sc_freq.J_pad[-1]  # TODO
        S_1_tm_T_hat = _pad_transpose_fft(S_1_list, total_height, B, B.rfft)

        if oversampling_fr == 'auto':
            # subsample as we would in min-padded case
            total_subsample_fr_max = sc_freq.J - max(sc_freq.j0s)
        else:
            # subsample regularly (relative to current padding)
            total_subsample_fr_max = sc_freq.J

        # Low-pass filtering over frequency
        k_fr_J = max(total_subsample_fr_max - oversampling, 0)
        S_1_fr_T_c = B.cdgmm(S_1_tm_T_hat, sc_freq.phi_f[0])
        # TODO if `total_height` here disagrees with later's, must change k_fr_J
        S_1_fr_T_hat = B.subsample_fourier(S_1_fr_T_c, 2**k_fr_J)
        S_1_fr_T = B.irfft(S_1_fr_T_hat)

        # unpad + transpose, append to out
        # TODO dedicated ind_start, ind_end
        if out_type == 'list':# or oversampling_fr != 'auto':
            S_1_fr_T = unpad(S_1_fr_T, sc_freq.ind_start[-1][k_fr_J],
                             sc_freq.ind_end[-1][k_fr_J])
        elif out_type == 'array':# and oversampling_fr == 'auto':
            S_1_fr_T = unpad(S_1_fr_T, sc_freq.ind_start_max[k_fr_J],
                             sc_freq.ind_end_max[k_fr_J])

        S_1_fr = B.transpose(S_1_fr_T)
        out_S_1.append({'coef': S_1_fr, 'j': (), 'n': (), 's': ()})
        # RFC: should we put placeholders for j1 and n1 instead of empty tuples?
    else:
        # TODO what to do here?
        out_S_1.append({'coef': [], 'j': (), 'n': (), 's': ()})

    ##########################################################################
    # Second order: separable convolutions (along time & freq), and low-pass
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        if j2 == 0:
            continue

        # Wavelet transform over time
        Y_2_arr = None
        for n1 in range(len(psi1)):
            # Retrieve first-order coefficient in the list
            j1 = psi1[n1]['j']
            if j1 >= j2:
                continue
            U_1_hat = U_1_hat_list[n1]

            # Convolution and downsampling
            k1 = max(j1 - oversampling, 0)       # what we subsampled in 1st-order
            k2 = max(j2 - k1 - oversampling, 0)  # what we subsample now in 2nd
            Y_2_c = B.cdgmm(U_1_hat, psi2[n2][k1])
            Y_2_hat = B.subsample_fourier(Y_2_c, 2**k2)
            Y_2_c = B.ifft(Y_2_hat)

            if Y_2_arr is None:
                if oversampling_fr == 'auto' and out_type == 'array':
                    pad_fr = sc_freq.J_pad[-1]
                else:
                    pad_fr = sc_freq.J_pad[n2]
                Y_2_arr = backend.zeros((2**pad_fr, Y_2_c.shape[-1]),
                                        dtype=Y_2_c.dtype)
            Y_2_arr[n1] = Y_2_c

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
    # Second order: `X * (phi_t * psi_f)`
    # take largest subsampling factor
    j2 = J - 1

    # Low-pass filtering over time, with filter length matching first-order's
    Y_2_arr = None
    for n1 in range(len(psi1)):
        j1 = psi1[n1]['j']
        if j1 >= j2:
            continue

        # Convolution and downsampling
        k1 = max(j1 - oversampling, 0)       # what we subsampled in 1st-order
        k2 = max(j2 - k1 - oversampling, 0)  # what we subsample now in 2nd
        Y_2_c = S_1_c_list[n1]               # reuse 1st-order U_1_hat * phi[k1]
        Y_2_hat = B.subsample_fourier(Y_2_c, 2**k2)
        Y_2_c = B.ifft(Y_2_hat)

        if Y_2_arr is None:
            pad_fr = sc_freq.J_pad[-1]
            Y_2_arr = backend.zeros((2**pad_fr, Y_2_c.shape[-1]),
                                    dtype=Y_2_c.dtype)
        Y_2_arr[n1] = Y_2_c

    # sum is same for all `n1`, just take last
    k1_plus_k2 = k1 + k2

    # swap axes & map to Fourier domain to prepare for conv along freq
    Y_2_hat = _transpose_fft(Y_2_arr, B, B.fft)

    # Transform over frequency + low-pass
    # `* psi_f` part of `X * (phi_t * psi_f)`
    _frequency_scattering(Y_2_hat, j2, -1, pad_fr, k1_plus_k2, commons,
                          out_S_2['phi_t * psi_f'], spin_down=False)

    ##########################################################################

    out_S = []
    out_S.extend(out_S_1)
    for outs in out_S_2.values():
        if isinstance(outs[0], list):
            for o in outs:
                out_S.extend(o)
        else:
            out_S.extend(outs)

    # if out_type == 'array':  # TODO breaks for first-order coeffs
    #     out_S = B.concatenate([x['coef'] for x in out_S])
    # elif out_type == 'list':  # TODO why pop? need for viz
    #     for x in out_S:
    #         x.pop('n')

    return out_S


def _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons, out_S_2,
                          spin_down=True):
    B, sc_freq, oversampling, oversampling_fr, out_type, *_ = commons

    psi1_fs = [sc_freq.psi1_f_up]
    if spin_down:
        psi1_fs.append(sc_freq.psi1_f_down)

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, psi1_f in enumerate(psi1_fs):
        for n1_fr in range(len(psi1_f)):
            # Wavelet transform over frequency
            # k0_fr accounts for sc_freq.J_pad
            if oversampling_fr == 'auto':
                # subsample as we would in min-padded case
                reference_subsample_equiv_due_to_pad = max(sc_freq.j0s)
            else:
                # subsample regularly (relative to current padding)
                reference_subsample_equiv_due_to_pad = sc_freq.j0s[n2]
            subsample_equiv_due_to_pad = max(sc_freq.J_pad) - pad_fr

            j1_fr = psi1_f[n1_fr]['j']
            k0_fr = reference_subsample_equiv_due_to_pad
            n1_fr_subsample = max(j1_fr - k0_fr - oversampling, 0)

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
    B, sc_freq, oversampling, oversampling_fr, out_type, *_ = commons

    subsample_equiv_due_to_pad = max(sc_freq.J_pad) - pad_fr
    j1_fr = sc_freq.psi1_f_up[-1]['j']   # take largest subsampling factor
    n1_fr_subsample = max(j1_fr - subsample_equiv_due_to_pad - oversampling, 0)

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
    (B, sc_freq, oversampling, oversampling_fr, out_type, unpad, J, phi,
     ind_start, ind_end, average) = commons

    if oversampling_fr == 'auto':
        # subsample as we would in min-padded case
        total_subsample_fr_max = sc_freq.J - max(sc_freq.j0s)
    else:
        # subsample regularly (relative to current padding)
        total_subsample_fr_max = sc_freq.J

    total_subsample_so_far = n1_fr_subsample + subsample_equiv_due_to_pad
    lowpass_subsample_fr = max(total_subsample_fr_max -
                               total_subsample_so_far - oversampling, 0)
    total_subsample_fr = total_subsample_so_far + lowpass_subsample_fr
    # greater idx = shorter phi_f, so increase if padding was less (greater j0)
    # or `*sc_freq.psi1_f` was subsampled more (greater n1_fr_subsample)

    if average:
        # Low-pass filtering over frequency
        # k1_fr_J = max(sc_freq.J - k1_fr - oversampling, 0)
        U_2_hat = B.rfft(U_2_m)
        S_2_fr_c = B.cdgmm(U_2_hat, sc_freq.phi_f[total_subsample_so_far])
        S_2_fr_hat = B.subsample_fourier(S_2_fr_c, 2**lowpass_subsample_fr)
        S_2_fr = B.irfft(S_2_fr_hat)

        # TODO unpad frequency domain iff out_type == "list"
        # TODO shouldn't we *always* unpad?
        if out_type == 'list':
            S_2_fr = unpad(S_2_fr,
                           sc_freq.ind_start[n2][total_subsample_fr],
                           sc_freq.ind_end[n2][total_subsample_fr])
        elif out_type == 'array':
            S_2_fr = unpad(S_2_fr,
                           sc_freq.ind_start_max[total_subsample_fr],
                           sc_freq.ind_end_max[total_subsample_fr])
        # Swap time and frequency subscripts again
        S_2_fr = B.transpose(S_2_fr)

        # Low-pass filtering over time
        k2_tm_J = max(J - k1_plus_k2 - oversampling, 0)
        U_2_hat = B.rfft(S_2_fr)
        S_2_c = B.cdgmm(U_2_hat, phi[k1_plus_k2])
        S_2_hat = B.subsample_fourier(S_2_c, 2**k2_tm_J)
        S_2_r = B.irfft(S_2_hat)

        S_2 = unpad(S_2_r, ind_start[k1_plus_k2 + k2_tm_J],
                    ind_end[k1_plus_k2 + k2_tm_J])
    else:
        if out_type == 'list':
            S_2_fr = unpad(S_2_fr,
                           sc_freq.ind_start[n2][total_subsample_fr],
                           sc_freq.ind_end[n2][total_subsample_fr])
        elif out_type == 'array':
            S_2_fr = unpad(S_2_fr,
                           sc_freq.ind_start_max[total_subsample_fr],
                           sc_freq.ind_end_max[total_subsample_fr])
        S_2_r = B.transpose(U_2_m)
        S_2 = unpad(S_2_r, ind_start[k1_plus_k2],
                    ind_end[k1_plus_k2])
    return S_2


def _pad_transpose_fft(coeff_list, total_height, B, fft):
    # zero-pad along frequency; enables convolving with longer low-pass
    padding_row = coeff_list[-1] * 0
    for i in range(total_height - len(coeff_list)):
        coeff_list.append(padding_row)

    # Concatenate along the frequency axis
    out = B.concatenate(coeff_list)
    # swap dims to convolve along frequency
    out = B.transpose(out)
    # Map to Fourier domain
    out = fft(out)
    return out


def _transpose_fft(coeff_arr, B, fft):
    # swap dims to convolve along frequency
    out = B.transpose(coeff_arr)
    # Map to Fourier domain
    out = fft(out)
    return out


__all__ = ['timefrequency_scattering']
