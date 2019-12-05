# Authors: Edouard Oyallon, Muawiz Chaudhary
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    empty_like = backend.empty_like

    order0_size = 1
    order1_size = L * J
    order2_size = L ** 2 * J * (J - 1) // 2
    output_size = order0_size + order1_size

    if max_order == 2:
        output_size += order2_size

    U_r = pad(x)

    U_0_c = fft(U_r, 'C2C')

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = fft(U_1_c, 'C2R', inverse=True)
    S_0 = unpad(S_0)

    output_shape = (x.shape[0], output_size) + S_0.shape[-2:]

    S = empty_like(x, output_shape)

    S[..., 0, :, :] = S_0.squeeze(-3)
    n_order1 = 1
    n_order2 = 1 + order1_size

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = fft(U_1_c, 'C2C', inverse=True)
        U_1_c = modulus(U_1_c)
        U_1_c = fft(U_1_c, 'C2C')

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi[j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        S_1_r = fft(S_1_c, 'C2R', inverse=True)
        S_1_r = unpad(S_1_r)

        S[..., n_order1, :, :] = S_1_r.squeeze(-3)
        n_order1 += 1

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            if j2 <= j1:
                continue
            U_2_c = cdgmm(U_1_c, psi[n2][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)
            U_2_c = fft(U_2_c, 'C2C')

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi[j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = fft(S_2_c, 'C2R', inverse=True)
            S_2_r = unpad(S_2_r)

            S[..., n_order2, :, :] = S_2_r.squeeze(-3)
            n_order2 += 1

    return S


__all__ = ['scattering2d']
