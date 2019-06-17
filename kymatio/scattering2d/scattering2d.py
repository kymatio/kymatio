# Authors: Edouard Oyallon
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna


__all__ = ['scattering2d']

def scattering2d(input, J, L, subsample_fourier, pad, modulus, fft, cdgmm, unpad, phi, psi, max_order, M_padded, N_padded):
    batch_shape = input.shape[:-2]
    signal_shape = input.shape[-2:]

    input = input.reshape((-1, 1) + signal_shape)

    order0_size = 1
    order1_size = L * J
    order2_size = L ** 2 * J * (J - 1) // 2
    output_size = order0_size + order1_size

    if max_order == 2:
        output_size += order2_size

    S = input.new(input.size(0),
                  input.size(1),
                  output_size,
                  M_padded//(2**J)-2,
                  N_padded//(2**J)-2)
    U_r = pad(input)
    U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c

    # First low pass filter
    U_1_c = subsample_fourier(cdgmm(U_0_c, phi[0]), k=2**J)

    U_J_r = fft(U_1_c, 'C2R')

    S[..., 0, :, :] = unpad(U_J_r)
    n_order1 = 1
    n_order2 = 1 + order1_size

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if(j1 > 0):
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = fft(U_1_c, 'C2C', inverse=True)
        U_1_c = fft(modulus(U_1_c), 'C2C')

        # Second low pass filter
        U_2_c = subsample_fourier(cdgmm(U_1_c, phi[j1]), k=2**(J-j1))
        U_J_r = fft(U_2_c, 'C2R')
        S[..., n_order1, :, :] = unpad(U_J_r)
        n_order1 += 1

        if max_order == 2:
            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = subsample_fourier(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2-j1))
                    U_2_c = fft(U_2_c, 'C2C', inverse=True)
                    U_2_c = fft(modulus(U_2_c), 'C2C')

                    # Third low pass filter
                    U_2_c = subsample_fourier(cdgmm(U_2_c, phi[j2]), k=2 ** (J-j2))
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n_order2, :, :] = unpad(U_J_r)
                    n_order2 += 1

    scattering_shape = S.shape[-3:]
    S = S.reshape(batch_shape + scattering_shape)

    return S
