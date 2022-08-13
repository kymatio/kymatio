def scattering1d_depthfirst(U0, backend, filters, oversampling, average_local):
    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    # Get S0
    phi = filters[0]
    log2_T = phi['j']
    k0 = max(log2_T - oversampling, 0)

    if average_local:
        S_0_c = cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = subsample_fourier(S_0_c, 2**k0)
        S_0_r = irfft(S_0_hat)
        # S_0_r and U_0 are 2D arrays indexed by (batch, time)
        yield {'coef': S_0_r, 'j': (), 'n': ()}
    else:
        yield {'coef': U_0, 'j': (), 'n': ()}

    # First order:
    psi1 = filters[1]
    U_1_hats = []
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        sub1_adj = min(j1, log2_T) if average_local else j1
        k1 = max(sub1_adj - oversampling, 0)
        U_1_c = cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)

        # Take the modulus
        U_1_m = modulus(U_1_c)

        if average_local:
            U_1_hat = rfft(U_1_m)
            # Depth-first algorithm: store U1 in anticipation of next layer
            U_1_hats.append({'coef': U_1_hat, 'j': (j1,), 'n': (n1,)})

        if average_local:
            # Convolve with phi_J
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi['levels'][k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = irfft(S_1_hat)
            # S_1_r and U_1_m are 2D arrays indexed by (batch, time)
            yield {'coef': S_1_r, 'j': (j1,), 'n': (n1,)}
        else:
            yield {'coef': U_1_m, 'j': (j1,), 'n': (n1,)}

    # Second order. Note that n2 is the outer loop in the depth-first algorithm
    psi2 = filters[2]
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        Y2 = []

        for U_1_hat in U_1_hats:
            j1 = U_1_hat['j'][0]

            if j2 > j1:
                sub2_adj = min(j2, log2_T) if average else j2
                k2 = max(sub2_adj - k1 - oversampling, 0)
                U_2_c = cdgmm(U_1_hat['coef'], psi2[n2]['levels'][k1])
                U_2_hat = subsample_fourier(U_2_c, 2**k2)
                U_2_c = ifft(U_2_hat)
                Y2.append(U_2_c)

        # Concatenate over the penultimate dimension with shared n2.
        # Y2 is a complex-valued 3D array indexed by (batch, n1, time)
        Y2 = concatenate(Y2, -2)

        # Y2 is a stack of multiple n1 paths so we put (-1) as placeholder.
        # In practice n1 ranges between 0 (included) and Y2.shape[-2] (excluded)
        # so there is no loss of information in this concatenation.
        yield {'coef': Y2, 'j': (-1, j2), 'n': (-1, n2)}
