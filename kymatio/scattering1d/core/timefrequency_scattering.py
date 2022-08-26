import numpy as np


def joint_timefrequency_scattering(U_0, backend, filters, oversampling,
         average_local, filters_fr, oversampling_fr, average_local_fr):
    # Zeroth order: S0(t[log_T]) if average_local, U0(t) otherwise
    time_gen = time_scattering_widthfirst(
        U_0, backend, filters, oversampling, average_local)
    yield next(time_gen)

    # First order: S1(n1, t) = (|x*psi_{n1}|*phi)(t[log2_T])
    S_1 = next(time_gen)

    # Y_fr_{n_fr}(n1, t[log2_T]) = (|x*psi_{n1}|*phi*psi_{n_fr})(t[log2_T])
    yield from frequency_scattering(S_1, backend, filters_fr, average_local_fr,
        oversampling_fr, spinned=False)

    # Second order: Y2_{n2}(n1, t) = (|x*psi_{n1}|*psi_{n2})(t[j2])
    for Y_2 in time_gen:

        # Y_fr_{n2,n_fr}(n1[j_fr], t[j2])
        #     = (|x*psi_{n1}|*psi_{n2}*psi_{n_fr})(t[j2])
        yield from frequency_scattering(Y_2, backend, filters_fr,
            average_local_fr, oversampling_fr, spinned=True)


def time_scattering_widthfirst(U_0, backend, filters, oversampling, average_local):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    oversampling : int >=0
        Yields scattering coefficients with a temporal stride equal to
        max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
        one unit halves the stride, until reaching a stride of 1.

    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.
    filters_fr : [phi, psis] list where
        * phi is a dictionary describing the low-pass filter of width F, used
          to average S1 and S2 in frequency if and only if average_local_fr.
        * psis is a list of dictionaries, each describing a low-pass or band-pass
          band-pass filter indexed by n_fr. The first element, n_fr=0, corresponds
          to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
          Other elements, such that n_fr>0, correspond to "spinned" band-pass
          filter, where spin denotes the sign of the center frequency xi.
    oversampling_fr : int >=0
        Yields joint time-frequency scattering coefficients with a frequential
        stride of max(1, 2**(log2_F-oversampling_fr)). Raising oversampling_fr
        by one halves the stride, until reaching a stride of 1.
    average_local_fr : boolean
        whether the result will be locally averaged with phi after this function

    Yields
    ------
    # Zeroth order
    if average_local:
        * S_0 indexed by (batch, time[log2_T])
    else:
        * U_0 indexed by (batch, time)

    # First order
    for n_fr < len(filters_fr):
        * Y_1_fr indexed by (batch, n1[n_fr], time[log2_T]), complex-valued,
            where n1 has been zero-padded to size N_fr before convolution

    # Second order
    for n2 < len(psi2):
        for n_fr < len(filters_fr):
            * Y_2_fr indexed by (batch, n1[n_fr], time[n2]), complex-valued,
                where n1 has been zero-padded to size N_fr before convolution

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    """
    # Zeroth order: S0(t[log_T]) if average_local, U0(t) otherwise
    time_gen = time_scattering_widthfirst(
        U_0, backend, filters, oversampling, average_local)
    yield next(time_gen)

    # First order: S1(n1, t) = (|x*psi_{n1}|*phi)(t[log2_T])
    S_1 = next(time_gen)

    # Y_1_fr_{n_fr}(n1, t[log2_T]) = (|x*psi_{n1}|*phi*psi_{n_fr})(t[log2_T])
    yield from frequency_scattering(S_1, backend,
        filters_fr, oversampling_fr, average_local_fr, spinned=False)

    # Second order: Y_2_{n2}(t[log2_T], n1[j_fr])
    #                   = (|x*psi_{n1}|*psi_{n2})(t[j2], n1[j_fr])
    for Y_2 in time_gen:

        # Y_2_fr_{n2,n_fr}(t[j2], n1[j_fr])
        #     = (|x*psi_{n1}|*psi_{n2}*psi_{n_fr})(t[j2], n1[j_fr])
        yield from frequency_scattering(Y_2, backend, filters_fr,
            oversampling_fr, average_local_fr, spinned=True)


def time_scattering_widthfirst(U_0, backend, filters, oversampling, average_local):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    oversampling : int >=0
        Yields scattering coefficients with a temporal stride equal to
        max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
        one unit halves the stride, until reaching a stride of 1.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.

    Yields
    ------
    if average_local:
        * S_0 indexed by (batch, time[log2_T])
    else:
        * U_0 indexed by (batch, time)
    * S_1 indexed by (batch, n1, time[log2_T])
    for n2 < len(psi2):
        * Y_2{n2} indexed by (batch, n1, time[j1]) and n1 s.t. j1 < j2

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (|x * psi_{n1}| * phi)(t), conv. over t, broadcast over n1
    Y_2{n2}(n1, t) = (U_1 * psi_{n2})(n1, t), conv. over t, broadcast over n1
    """
    # compute the Fourier transform
    U_0_hat = backend.rfft(U_0)

    # Get S0, a 2D array indexed by (batch, time)
    phi = filters[0]
    log2_T = phi['j']
    if average_local:
        k0 = max(log2_T - oversampling, 0)
        S_0_c = backend.cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = backend.subsample_fourier(S_0_c, 2 ** k0)
        S_0_r = backend.irfft(S_0_hat)
        yield {'coef': S_0_r, 'j': (), 'n': ()}
    else:
        yield {'coef': U_0, 'j': (), 'n': ()}

    # First order:
    psi1 = filters[1]
    U_1_hats = []
    S_1_list = []
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

        # Convolve with phi_J
        k1_J = max(log2_T - k1 - oversampling, 0)
        S_1_c = backend.cdgmm(U_1_hat, phi['levels'][k1])
        S_1_hat = backend.subsample_fourier(S_1_c, 2 ** k1_J)
        S_1_r = backend.irfft(S_1_hat)
        S_1_list.append(S_1_r)

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


def frequency_scattering(X, backend, filters_fr, oversampling_fr,
        average_local_fr, spinned):
    """
    Parameters
    ----------
    X : dictionary with keys 'coef' and 'n1_max'
        if spinned, X['coef']=Y_2 is complex-valued and indexed by
        (batch, n1, time[j2]), for some fixed n2 and variable n1 s.t. j1 < j2.
        else, X['coef']=S_1 is real-valued and indexed by
        (batch, n1, time[log2_T]) for variable n1 < len(psi1_f)
    backend : module
    filters_fr : [phi, psis] list where
        * phi is a dictionary describing the low-pass filter of width F, used
          to average S1 and S2 in frequency if and only if average_local_fr.
        * psis is a list of dictionaries, each describing a low-pass or band-pass
          band-pass filter indexed by n_fr. The first element, n_fr=0, corresponds
          to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
          Other elements, such that n_fr>0, correspond to "spinned" band-pass
          filter, where spin denotes the sign of the center frequency xi.
    oversampling_fr : int >=0
        Yields joint time-frequency scattering coefficients with a frequential
        stride max(1, 2**(log2_F-oversampling_fr)). Raising oversampling_fr
        by one halves the stride, until reaching a stride of 1.
    average_local_fr : boolean
        whether the result will be locally averaged with phi after this function
    spinned: boolean
        if True (complex input), yields Y_fr for all n_fr
        else (real input), yields Y_fr for only those n_fr s.t. spin>=0

    Yields
    ------
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    """

    # Unpack filters_fr list
    phi, psis = filters_fr
    log2_F = phi['j']

    # Swap time and frequency axis
    X_T = backend.swap_time_frequency(X['coef'])

    # Zero-pad frequency domain
    pad_right = phi['N'] - X['n1_max']
    X_pad = backend.pad_frequency(X_T, pad_right)

    # Spinned case switch
    if spinned:
        # Complex-input FFT
        X_hat = backend.cfft(X_pad)
    else:
        # Real-input FFT
        X_hat = backend.rfft(X_pad)
        # Restrict to nonnegative spins
        psis = filter(lambda psi: psi['xi']>=0, psis)

    for n_fr, psi in enumerate(psis):
        j_fr = psi['j']
        spin = np.sign(psi['xi'])
        sub_fr_adj = min(j_fr, log2_F) if average_local_fr else j_fr
        k_fr = max(sub_fr_adj - oversampling_fr, 0)
        Y_fr_hat = backend.cdgmm(X_hat, psi['levels'][0])
        Y_fr_sub = backend.subsample_fourier(Y_fr_hat, 2 ** k_fr)
        Y_fr = backend.ifft(Y_fr_sub)
        Y_fr = backend.swap_time_frequency(Y_fr)
        # If not spinned, X['n']=S1['n']=(-1,) is a 1-tuple.
        # If spinned, X['n']=Y2['n']=(-1,n2) is a 2-tuple.
        # In either case, we elide the "-1" placeholder and define the new 'n'
        # as (X['n'][1:] + (n_fr,)), i.e., n=(n_fr,) is not spinned
        # and n=(n2, n_fr) if spinned. This 'n' tuple is unique.
        yield {**X, 'coef': Y_fr, 'n': (X['n'][1:] + (n_fr,)),
            'j_fr': (j_fr,), 'n_fr': (n_fr,), 'spin': spin}


def time_averaging(U_2, backend, phi_f, oversampling):
    """
    Parameters
    ----------
    U_2 : dictionary with keys 'coef' and 'j', typically returned by
        frequency_scattering
    backend : module
    phi_f : dictionary. Temporal low-pass filter in Fourier domain,
        same as scattering1d
    oversampling : int >=0
        Yields scattering coefficients with a temporal stride equal to
        max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
        one unit halves the stride, until reaching a stride of 1.

    Returns
    -------
    S_2{n2,n_fr} indexed by (batch, n1[n_fr], time[log2_T])

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    U_2{n2,n_fr}(t, n1) = |Y_2_fr{n2,n_fr}|(t, n1)
    """
    log2_T = phi_f['j']
    k_in = U_2['j'][-1]
    k_J = max(log2_T - k_in - oversampling, 0)
    U_hat = backend.rfft(U_2['coef'])
    S_c = backend.cdgmm(U_hat, phi_f['levels'][k_in])
    S_hat = backend.subsample_fourier(S_c, 2 ** k_J)
    S_2 = backend.irfft(S_hat)
    return {**U_2, 'coef': S_2}


def frequency_averaging(U_2, backend, phi_fr_f, oversampling_fr, average_fr):
    """
    Parameters
    ----------
    U_2 : dictionary with keys 'coef' and 'j_fr', typically returned by
        frequency_scattering or time_averaging
    backend : module
    phi_fr_f : dictionary. Frequential low-pass filter in Fourier domain.
    oversampling_fr : int >=0
        Yields joint time-frequency scattering coefficients with a frequential
        stride max(1, 2**(log2_F-oversampling_fr)). Raising oversampling_fr
        by one halves the stride, until reaching a stride of 1.
    average_fr : string
        Either 'local', 'global', or False.

    Returns
    -------
    if average_fr == 'local':
        * S_2{n2,n_fr} indexed by (batch, n1[log2_F], time[j2])
    if average_fr == 'global':
        * S_2{n2,n_fr} indexed by (batch, 1, time[j2])
    if average_fr == False:
        * S_2{n2,n_fr} indexed by (batch, n1[j_fr], time[j2])

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    U_2{n2,n_fr}(t, n1) = |Y_2_fr{n2,n_fr}|(t[j2], n1[j_fr])
    """
    if average_fr:
        log2_F = phi_fr_f['j']
        U_2_T = backend.swap_time_frequency(U_2['coef'])
        if average_fr == 'global':
            S_2_T = backend.average_global(U_2_T)
            n1_stride = U_2['n1_max']
        elif average_fr == 'local':
            k_in = U_2['j_fr'][-1]
            k_J = max(log2_F - k_in - oversampling_fr, 0)
            U_hat = backend.rfft(U_2_T)
            S_c = backend.cdgmm(U_hat, phi_fr_f['levels'][k_in])
            S_hat = backend.subsample_fourier(S_c, 2 ** k_J)
            S_2_T = backend.irfft(S_hat)
            n1_stride = 2**max(log2_F - oversampling_fr, 0)
        S_2 = backend.swap_time_frequency(S_2_T)
        return {**U_2, 'coef': S_2, 'n1_stride': n1_stride}
    elif not average_fr:
        n1_stride = 2**max(U_2['j_fr'][-1] - oversampling_fr, 0)
        return {**U_2, 'n1_stride': n1_stride}
