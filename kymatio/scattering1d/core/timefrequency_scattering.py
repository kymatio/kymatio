
def timefrequency_scattering(
        x, pad, unpad, backend, J, psi1, psi2, phi, sc_freq,
        pad_left=0, pad_right=0, ind_start=None, ind_end=None, oversampling=0,
        max_order=2, average=True, size_scattering=(0, 0, 0), out_type='array'):
    """
    Main function implementing the joint time-frequency scattering transform.
    """
    transpose = backend.transpose
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    batch_size = x.shape[0]
    kJ = max(J - oversampling, 0)
    temporal_size = ind_end[kJ] - ind_start[kJ]
    out_S_1, out_S_2 = [], [[], []]

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)

    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    # First order:
    U_1_hat_list = []
    S_1_list = []
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)
        U_1_c = cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)

        # Modulus
        U_1_m = modulus(U_1_c)

        # Map to Fourier domain
        U_1_hat = rfft(U_1_m)
        U_1_hat_list.append(U_1_hat)

        # Apply low-pass filtering over time (optional) and unpad
        if average:
            # Low-pass filtering over time
            k1_J = max(J - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi[k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = irfft(S_1_hat)

            # Unpad
            S_1_tm = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
            S_1_list.append(S_1_tm)
        else:
            # Unpad
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])
            out_S_1.append({'coef': S_1, 'j': (j1,), 'n': (n1,)})

    # Apply low-pass filtering over frequency (optional) and unpad
    if average:
        # zero-pad along frequency; enables convolving with longer low-pass
        total_height = 2 ** sc_freq.J_pad
        padding_row = 0 * S_1_tm
        for i in range(total_height - len(S_1_list)):
            if i % 2 == 0:
                S_1_list.insert(0, padding_row)
            else:
                S_1_list.append(padding_row)
        S_1_tm = concatenate(S_1_list)

        # swap dims to convolve along frequency
        S_1_tm_T = transpose(S_1_tm)

        # Low-pass filtering over frequency
        k_fr_J = max(sc_freq.J - oversampling, 0)
        S_1_tm_T_hat = rfft(S_1_tm_T)
        S_1_fr_T_c = cdgmm(S_1_tm_T_hat, sc_freq.phi_f[0])
        # TODO if `total_height` here disagrees with later's, must change k_fr_J
        S_1_fr_T_hat = subsample_fourier(S_1_fr_T_c, 2**k_fr_J)
        S_1_fr_T = irfft(S_1_fr_T_hat)

        # unpad + transpose, append to out
        if out_type == 'list':  # TODO
            S_1_fr_T = unpad(S_1_fr_T, sc_freq.ind_start[k_fr_J],
                             sc_freq.ind_end[k_fr_J])
        S_1_fr = transpose(S_1_fr_T)
        out_S_1.append({'coef': S_1_fr, 'j': (), 'n': ()})
        # RFC: should we put placeholders for j1 and n1 instead of empty tuples?


    total_height = 2 ** sc_freq.J_pad
    # Second order: separable convolutions (along time & freq), and low-pass
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        if j2 == 0:
            continue

        # Wavelet transform over time
        Y_2_list = []
        for n1 in range(len(psi1)):
            # Retrieve first-order coefficient in the list
            j1 = psi1[n1]['j']
            if j1 >= j2:
                continue
            U_1_hat = U_1_hat_list[n1]

            # Convolution and downsampling
            k1 = max(j1 - oversampling, 0)       # what we subsampled in 1st-order
            k2 = max(j2 - k1 - oversampling, 0)  # what we subsample now in 2nd
            Y_2_c = cdgmm(U_1_hat, psi2[n2][k1])
            Y_2_hat = subsample_fourier(Y_2_c, 2**k2)
            Y_2_list.append(ifft(Y_2_hat))

        # sum is same for all `n1`, just take last
        k1_plus_k2 = k1 + k2

        # zero-pad along frequency; enables convolving with longer low-pass
        padding_row = Y_2_hat * 0
        for i in range(total_height - len(Y_2_list)):
            if i % 2 == 0:
                Y_2_list.insert(0, padding_row)
            else:
                Y_2_list.append(padding_row)

        # Concatenate along the frequency axis
        Y_2 = concatenate(Y_2_list)

        # Swap time and frequency subscripts to prepare for frequency scattering
        Y_2_T = transpose(Y_2)

        # Complex FFT is not implemented in the backend, only RFFT and IFFT
        # so we use IFFT which is equivalent up to conjugation.
        Y_2_hat = fft(Y_2_T)

        # Transform over frequency + low-pass, for both spins
        for s1_fr, psi1_f in enumerate([sc_freq.psi1_f_up, sc_freq.psi1_f_down]):
            for n1_fr in range(len(psi1_f)):
                # Wavelet transform over frequency
                j1_fr = psi1_f[n1_fr]['j']
                k1_fr = max(j1_fr - oversampling, 0)
                Y_fr_c = cdgmm(Y_2_hat, psi1_f[n1_fr][0])
                Y_fr_hat = subsample_fourier(Y_fr_c, 2**k1_fr)
                Y_fr_c = ifft(Y_fr_hat)

                # Modulus
                U_2_m = modulus(Y_fr_c)

                # convolve by Phi = phi_t * phi_f
                if average:
                    # Low-pass filtering over frequency
                    k1_fr_J = max(sc_freq.J - k1_fr - oversampling, 0)
                    U_2_hat = rfft(U_2_m)
                    S_2_fr_c = cdgmm(U_2_hat, sc_freq.phi_f[k1_fr])
                    S_2_fr_hat = subsample_fourier(S_2_fr_c, 2**k1_fr_J)
                    S_2_fr = irfft(S_2_fr_hat)

                    # TODO unpad frequency domain iff out_type == "list"
                    # TODO shouldn't we *always* unpad?
                    if out_type == 'list':
                        S_2_fr = unpad(S_2_fr, sc_freq.ind_start[k1_fr_J + k1_fr],
                                       sc_freq.ind_end[k1_fr_J + k1_fr])

                    # Swap time and frequency subscripts again
                    S_2_fr = transpose(S_2_fr)

                    # Low-pass filtering over time
                    k2_tm_J = max(J - k1_plus_k2 - oversampling, 0)
                    U_2_hat = rfft(S_2_fr)
                    S_2_c = cdgmm(U_2_hat, phi[k1_plus_k2])
                    S_2_hat = subsample_fourier(S_2_c, 2**k2_tm_J)
                    S_2_r = irfft(S_2_hat)

                    S_2 = unpad(S_2_r, ind_start[k1_plus_k2 + k2_tm_J],
                                ind_end[k1_plus_k2 + k2_tm_J])
                else:
                    S_2_r = transpose(U_2_m)
                    S_2 = unpad(S_2_r, ind_start[k1_plus_k2],
                                ind_end[k1_plus_k2])

                spin = (1, -1)[s1_fr]
                out_S_2[s1_fr].append({'coef': S_2,
                                       'j': (j2, j1_fr),
                                       'n': (n2, n1_fr),
                                       's': spin})

    out_S = []
    out_S.extend(out_S_1)
    for o in out_S_2:
        out_S.extend(o)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])
    elif out_type == 'list':
        for x in out_S:
            x.pop('n')

    return out_S


__all__ = ['timefrequency_scattering']
