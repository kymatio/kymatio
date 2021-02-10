
def scattering3d_standard(x, pad, unpad, backend, J, L, phi, psi, max_order):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    irfft = backend.irfft
    ifft = backend.ifft
    cdgmm3d = backend.cdgmm3d
    concatenate = backend.concatenate
    subsample_fourier = backend.subsample_fourier
    concatenate_3d  = backend.concatenate_3d

    order0_size = 1
    order1_size = L * J
    order2_size = L ** 2 * J * (J - 1) // 2
    output_size = order0_size + order1_size

    if max_order == 2:
        output_size += order2_size

    out_S_0, out_S_1, out_S_2 = [], [], []
    U_r = pad(x)

    U_0_c = rfft(U_r)

    # First low pass filter
    U_1_c = cdgmm3d(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0)

    out_S_0.append(S_0)

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm3d(U_0_c, psi[n1][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)

        # Second low pass filter
        S_1_c = cdgmm3d(U_1_c, phi[j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        S_1_r = irfft(S_1_c)
        S_1_r = unpad(S_1_r)

        out_S_1.append(S_1_r)

        if max_order < 2:
            continue 
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            if j2 <= j1:
                continue
            U_2_c = cdgmm3d(U_1_c, psi[n2][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            S_2_c = cdgmm3d(U_2_c, phi[j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r)
            
            out_S_2.append(S_2_r)

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    out_S = concatenate_3d(out_S)
    return out_S


