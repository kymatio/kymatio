import numpy as np


def joint_timefrequency_scattering(U_0, backend, filters, log2_stride,
         average_local, filters_fr, log2_stride_fr, average_local_fr):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    log2_stride : int >=0
        Yields coefficients with a temporal stride equal to 2**log2_stride.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.
    filters_fr : [phi, psis] list where
     * phi is a dictionary describing the low-pass filter of width F, used
       to average S1 and S2 in frequency if and only if average_local_fr.
     * psis is a list of dictionaries, each describing a low-pass or band-pass
       filter indexed by n_fr. The first element, n_fr=0, corresponds
       to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
       Other elements, such that n_fr>0, correspond to "spinned" band-pass
       filter, where spin denotes the sign of the center frequency xi.
    log2_stride_fr : int >=0
        yields coefficients with a frequential stride of 2**log2_stride_fr
        if average_local_fr (see below), otherwise 2**j_fr
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
    # Zeroth order: S0(t[log2_T]) if average_local, U0(t) otherwise
    time_gen = time_scattering_widthfirst(
        U_0, backend, filters, log2_stride, average_local)
    yield next(time_gen)

    # First order: S1(n1, t) = (|x*psi_{n1}|*phi)(t[log2_T])
    S_1 = next(time_gen)

    # Y_1_fr_{n_fr}(n1, t[log2_T]) = (|x*psi_{n1}|*phi*psi_{n_fr})(t[log2_T])
    yield from frequency_scattering(S_1, backend,
        filters_fr, log2_stride_fr, average_local_fr, spinned=False)

    # Second order: Y_2_{n2}(t[log2_T], n1[j_fr])
    #                   = (|x*psi_{n1}|*psi_{n2})(t[j2], n1[j_fr])
    for Y_2 in time_gen:

        # Y_2_fr_{n2,n_fr}(t[j2], n1[j_fr])
        #     = (|x*psi_{n1}|*psi_{n2}*psi_{n_fr})(t[j2], n1[j_fr])
        yield from frequency_scattering(Y_2, backend, filters_fr,
            log2_stride_fr, average_local_fr, spinned=True)


def time_scattering_widthfirst(U_0, backend, filters, log2_stride, average_local):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    log2_stride : int >=0
        Yields coefficients with a temporal stride equal to 2**log2_stride.
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
        k0 = log2_stride
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
        k1 = min(j1, log2_stride) if average_local else j1
        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2 ** k1)
        U_1_c = backend.ifft(U_1_hat)

        # Take the modulus
        U_1_m = backend.modulus(U_1_c)

        U_1_hat = backend.rfft(U_1_m)
        # Width-first algorithm: store U1 in anticipation of next layer
        U_1_hats.append({'coef': U_1_hat, 'j': (j1,), 'n': (n1,)})

        # Convolve with phi_J
        k1_J = log2_stride - k1
        S_1_c = backend.cdgmm(U_1_hat, phi['levels'][k1])
        S_1_hat = backend.subsample_fourier(S_1_c, 2 ** k1_J)
        S_1_r = backend.irfft(S_1_hat)
        S_1_list.append(S_1_r)

    # Stack S1 paths over the penultimate dimension with shared n1.
    # S1 is a real-valued 3D array indexed by (batch, n1, time)
    S_1 = backend.stack(S_1_list, 2)

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
            k1 = min(j1, log2_stride) if average_local else j1

            if j2 > j1:
                k2 = min(j2-k1, log2_stride) if average_local else (j2-k1)
                U_2_c = backend.cdgmm(U_1_hat['coef'], psi2[n2]['levels'][k1])
                U_2_hat = backend.subsample_fourier(U_2_c, 2 ** k2)
                U_2_c = backend.ifft(U_2_hat)
                Y_2_list.append(U_2_c)

        if len(Y_2_list) > 0:
            # Stack over the penultimate dimension with shared n2.
            # Y_2 is a complex-valued 3D array indexed by (batch, n1, time)
            Y_2 = backend.stack(Y_2_list, 2)

            # Y_2 is a stack of multiple n1 paths so we put (-1) as placeholder.
            # n1 ranges between 0 (included) and n1_max (excluded), which we store
            # separately for the sake of meta() and padding/unpadding downstream.
            yield {'coef': Y_2, 'j': (-1, j2), 'n': (-1, n2), 'n1_max': len(Y_2_list)}


def frequency_scattering(X, backend, filters_fr, log2_stride_fr,
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
          filter indexed by n_fr. The first element, n_fr=0, corresponds
          to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
          Other elements, such that n_fr>0, correspond to "spinned" band-pass
          filter, where spin denotes the sign of the center frequency xi.
    log2_stride_fr : int >=0
        yields coefficients with a frequential stride of 2**log2_stride_fr
        if average_local_fr (see below), otherwise 2**j_fr
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
        k_fr = min(j_fr, log2_stride_fr) if average_local_fr else j_fr
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
            'j_fr': (j_fr,), 'n_fr': (n_fr,), 'n1_stride': (2**j_fr),
            'spin': spin}


def time_averaging(U_2, backend, phi_f, log2_stride):
    """
    Parameters
    ----------
    U_2 : dictionary with keys 'coef' and 'j', typically returned by
        frequency_scattering
    backend : module
    phi_f : dictionary. Temporal low-pass filter in Fourier domain,
        same as scattering1d
    log2_stride : int >=0
        Yields coefficients with a temporal stride equal to 2**log2_stride.

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
    k_J = log2_stride - k_in
    U_hat = backend.rfft(U_2['coef'])
    S_c = backend.cdgmm(U_hat, phi_f['levels'][k_in])
    S_hat = backend.subsample_fourier(S_c, 2 ** k_J)
    S_2 = backend.irfft(S_hat)
    return {**U_2, 'coef': S_2}


def frequency_averaging(U_2, backend, phi_fr_f, log2_stride_fr, average_fr):
    """
    Parameters
    ----------
    U_2 : dictionary with keys 'coef' and 'j_fr', typically returned by
        frequency_scattering or time_averaging
    backend : module
    phi_fr_f : dictionary. Frequential low-pass filter in Fourier domain.
    log2_stride_fr : int >=0
        yields coefficients with a frequential stride of 2**log2_stride_fr
        if average_local_fr (see below), otherwise 2**j_fr
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
            k_in = min(U_2['j_fr'][-1], log2_F)
            k_J = log2_stride_fr - k_in
            U_hat = backend.rfft(U_2_T)
            S_c = backend.cdgmm(U_hat, phi_fr_f['levels'][k_in])
            S_hat = backend.subsample_fourier(S_c, 2 ** k_J)
            S_2_T = backend.irfft(S_hat)
            n1_stride = 2**log2_stride_fr
        S_2 = backend.swap_time_frequency(S_2_T)
        return {**U_2, 'coef': S_2, 'n1_stride': n1_stride}
    elif not average_fr:
        n1_stride = 2**max(U_2['j_fr'][-1], 0)
        return {**U_2, 'n1_stride': n1_stride}


def time_formatting(path, backend):
    if path["coef"] is None:
        # special case: user called meta(), so we propagate None
        coef_list = [None] * (1 + path["n1_max"] // path["n1_stride"])
    else:
        coef_list = backend.split_frequency_axis(path["coef"])
    for i, n1 in enumerate(range(0, path["n1_max"], path["n1_stride"])):
        split_path = {**path, "coef": coef_list[i], "order": len(path["n"])-1}
        # If not spinned, X['n']=S1['n']=(n_fr,) is a 1-tuple.
        # If spinned, X['n']=Y2['n']=(n2,n_fr) is a 2-tuple.
        # In either case, we prepend n1 and define the new 'n'
        # as (n1,) + X['n'], i.e., n=(n1, n_fr) is not spinned
        # and n=(n1, n2, n_fr) if spinned. This 'n' tuple is unique.
        split_path["n"] = (n1,) + split_path["n"]
        del split_path["n1_max"]
        del split_path["n1_stride"]
        yield split_path


def jtfs_average_and_format(U_gen, backend, phi_f, log2_stride, average,
        phi_fr_f, oversampling_fr, average_fr, out_type, format):
    # Zeroth order
    path = next(U_gen)
    log2_T = phi_f['j']
    if average == 'global':
        path['coef'] = backend.average_global(path['coef'])
    yield {**path, 'order': 0}

    # First and second order
    for path in U_gen:
        # "The phase of the integrand must be set to a constant. This
        # freedom in setting the stationary phase to an arbitrary constant
        # value suggests the existence of a gauge boson" — Glinsky
        path['coef'] = backend.modulus(path['coef'])

        # Temporal averaging. Switch cases:
        # 1. If averaging is global, no need for unpadding at all.
        # 2. If averaging is local, averaging depends on order:
        #     2a. at order 1, U_gen yields
        #               Y_1_fr = S_1 * psi_{n_fr}
        #         no need for further averaging
        #     2b. at order 2, U_gen yields
        #               Y_2_fr = U_1 * psi_{n2} * psi_{n_fr}
        #         average with phi
        # (for simplicity, we assume oversampling=0 in the rationale above,
        #  but the implementation below works for any value of oversampling)
        if average == 'global':
            # Case 1.
            path['coef'] = backend.average_global(path['coef'])
        elif average == 'local' and len(path['n']) > 1:
            # Case 2b.
            path = time_averaging(path, backend, phi_f, log2_stride)

        # Frequential averaging. NB. if spin==0, U_gen is already averaged.
        # Hence, we only average if spin!=0, i.e. psi_{n_fr} in path.
        if average_fr and not path['spin'] == 0:
            path = frequency_averaging(path, backend,
                phi_fr_f, oversampling_fr, average_fr)

        # Frequential unpadding.
        if not (out_type == 'array' and format == 'joint'):
            path['coef'] = backend.unpad_frequency(
                path['coef'], path['n1_max'], path['n1_stride'])

        # Splitting and reshaping
        if format == 'joint':
            yield {**path, 'order': len(path['n'])}
        elif format == 'time':
            yield from time_formatting(path, backend)
