def scattering1d_widthfirst(U_0, backend, filters, oversampling, average_local):
    """
    Inputs
    ------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : (phi, psi1, psi2) tuple of dictionaries. same as scattering1d
    oversampling : int >= 0. same as scattering1d
    average_local : bool. same as scattering1d

    Yields
    ------
    if average_local:
        * S_0 indexed by (batch, time[log2_T])
        * S_1 indexed by (batch, n1, time[log2_T])
        * Y_2[n2=0] indexed by (batch, n1, time[log2_T]) and n1 s.t. j1 < j2
        * etc. for every n2 < len(psi2)
    else:
        * U_0 indexed by (batch, time)
        * U_1[n1=0] indexed by (batch, time[n1])
        * etc. for every n1 < len(psi1)
        * Y_2[n2=0] indexed by (batch, n1, time[n2]) and n1 s.t. j1 < j2
        * etc. for every n2 < len(psi2)

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1[n1](t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (|x * psi_{n1}| * phi)(t) broadcast over n1
    Y_2[n2](n1, t) = (U1 * psi_{n2})(n1, t) broadcast over n1
    """
    # compute the Fourier transform
    U_0_hat = backend.rfft(U_0)

    # Get S0
    phi = filters[0]
    log2_T = phi['j']
    k0 = max(log2_T - oversampling, 0)

    if average_local:
        S_0_c = backend.cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = backend.subsample_fourier(S_0_c, 2 ** k0)
        S_0_r = backend.irfft(S_0_hat)
        S_1_list = []
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
        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2 ** k1)
        U_1_c = backend.ifft(U_1_hat)

        # Take the modulus
        U_1_m = backend.modulus(U_1_c)

        U_1_hat = backend.rfft(U_1_m)
        # Width-first algorithm: store U1 in anticipation of next layer
        U_1_hats.append({'coef': U_1_hat, 'j': (j1,), 'n': (n1,)})

        if average_local:
            # Convolve with phi_J
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_c = backend.cdgmm(U_1_hat, phi['levels'][k1])
            S_1_hat = backend.subsample_fourier(S_1_c, 2 ** k1_J)
            S_1_r = backend.irfft(S_1_hat)
            S_1_list.append(S_1_r)
        else:
            # U_1_m is a 2D array indexed by (batch, time)
            yield {'coef': U_1_m, 'j': (j1,), 'n': (n1,)}

    if average_local:
        # Concatenate S1 paths over the penultimate dimension with shared n1.
        # S1 is a real-valued 3D array indexed by (batch, n1, time)
        S_1 = backend.concatenate(S_1_list)

        # S1 is a stack of multiple n1 paths so we put (-1) as placeholder.
        # n1 ranges between 0 (included) and n1_max (excluded), which we store
        # separately for the sake of meta() and padding/unpadding downstream.
        yield {'coef': S_1, 'j': (-1,), 'n': (-1,), 'n1_max': len(S_1_list)}

    # Second order. Note that n2 is the outer loop (width-first algorithm)
    psi2 = filters[2]
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        Y_2_list = []

        for U_1_hat in U_1_hats:
            j1 = U_1_hat['j'][0]
            sub1_adj = min(j1, log2_T) if average_local else j1
            k1 = max(sub1_adj - oversampling, 0)

            if j2 > j1:
                sub2_adj = min(j2, log2_T) if average_local else j2
                k2 = max(sub2_adj - k1 - oversampling, 0)
                U_2_c = backend.cdgmm(U_1_hat['coef'], psi2[n2]['levels'][k1])
                U_2_hat = backend.subsample_fourier(U_2_c, 2 ** k2)
                U_2_c = backend.ifft(U_2_hat)
                Y_2_list.append(U_2_c)

        if len(Y_2_list) > 0:
            # Concatenate over the penultimate dimension with shared n2.
            # Y_2 is a complex-valued 3D array indexed by (batch, n1, time)
            Y_2 = backend.concatenate(Y_2_list)

            # Y_2 is a stack of multiple n1 paths so we put (-1) as placeholder.
            # n1 ranges between 0 (included) and n1_max (excluded), which we store
            # separately for the sake of meta() and padding/unpadding downstream.
            yield {'coef': Y_2, 'j': (-1, j2), 'n': (-1, n2), 'n1_max': len(Y_2_list)}
