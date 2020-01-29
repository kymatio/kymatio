# Authors: Mathieu Andreux, Joakim Anden, Edouard Oyallon
# Scientific Ancestry: Joakim Anden, Mathieu Andreux, Vincent Lostanlen


def scattering1d(x, pad, unpad, backend, J, psi1, psi2, phi, pad_left=0,
        pad_right=0, ind_start=None, ind_end=None, oversampling=0,
        max_order=2, average=True, size_scattering=(0, 0, 0),
        vectorize=False, out_type='array'):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    x : Tensor
        a torch Tensor of size `(B, 1, T)` where `T` is the temporal size
    psi1 : dictionary
        a dictionary of filters (in the Fourier domain), with keys (`j`, `q`).
        `j` corresponds to the downsampling factor for
        :math:`x \\ast psi1[(j, q)]``, and `q` corresponds to a pitch class
        (chroma).
        * psi1[(j, n)] is itself a dictionary, with keys corresponding to the
        dilation factors: psi1[(j, n)][j2] corresponds to a support of size
        :math:`2^{J_\\text{max} - j_2}`, where :math:`J_\\text{max}` has been
        defined a priori (`J_max = size` of the padding support of the input)
        * psi1[(j, n)] only has real values;
        the tensors are complex so that broadcasting applies
    psi2 : dictionary
        a dictionary of filters, with keys (j2, n2). Same remarks as for psi1
    phi : dictionary
        a dictionary of filters of scale :math:`2^J` with keys (`j`)
        where :math:`2^j` is the downsampling factor.
        The array `phi[j]` is a real-valued filter.
    J : int
        scale of the scattering
    pad_left : int, optional
        how much to pad the signal on the left. Defaults to `0`
    pad_right : int, optional
        how much to pad the signal on the right. Defaults to `0`
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start
    oversampling : int, optional
        how much to oversample the scattering (with respect to :math:`2^J`):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to `0`
    order2 : boolean, optional
        Whether to compute the 2nd order or not. Defaults to `False`.
    average_U1 : boolean, optional
        whether to average the first order vector. Defaults to `True`
    size_scattering : tuple
        Contains the number of channels of the scattering, precomputed for
        speed-up. Defaults to `(0, 0, 0)`.
    vectorize : boolean, optional
        whether to return a dictionary or a tensor. Defaults to False.

    """
    subsample_fourier = backend.subsample_fourier
    modulus_complex = backend.modulus_complex
    real = backend.real
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate


    # S is simply a dictionary if we do not perform the averaging...
    batch_size = x.shape[0]
    kJ = max(J - oversampling, 0)
    temporal_size = ind_end[kJ] - ind_start[kJ]
    out_S_0, out_S_1, out_S_2 = [], [], []

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)

    # compute the Fourier transform
    U_0_hat = fft(U_0, 'C2C')

    # Get S0
    k0 = max(J - oversampling, 0)

    if average:
        S_0_c = cdgmm(U_0_hat, phi[0])
        S_0_hat = subsample_fourier(S_0_c, 2**k0)
        S_0_r = fft(S_0_hat, 'C2R', inverse=True)

        S_0 = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': ()})

    # First order:
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']

        k1 = max(j1 - oversampling, 0)

        assert psi1[n1]['xi'] < 0.5 / (2**k1)

        U_1_c = cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = fft(U_1_hat, 'C2C', inverse=True)

        # Take the modulus
        U_1_m = modulus_complex(U_1_c)

        if average or max_order > 1:
            U_1_hat = fft(U_1_m, 'C2C')

        if average:
            # Convolve with phi_J
            k1_J = max(J - k1 - oversampling, 0)

            S_1_c = cdgmm(U_1_hat, phi[k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = fft(S_1_hat, 'C2R', inverse=True)

            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            # just take the real value and unpad
            U_1_r = real(U_1_m)

            S_1 = unpad(U_1_r, ind_start[k1], ind_end[k1])

        out_S_1.append({'coef': S_1,
                        'j': (j1,),
                        'n': (n1,)})

        if max_order == 2:
            # 2nd order
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']

                if j2 > j1:
                    assert psi2[n2]['xi'] < psi1[n1]['xi']

                    # convolution + downsampling
                    k2 = max(j2 - k1 - oversampling, 0)

                    U_2_c = cdgmm(U_1_hat, psi2[n2][k1])
                    U_2_hat = subsample_fourier(U_2_c, 2**k2)
                    # take the modulus
                    U_2_c = fft(U_2_hat, 'C2C', inverse=True)

                    U_2_m = modulus_complex(U_2_c)

                    if average:
                        U_2_hat = fft(U_2_m, 'C2C')

                        # Convolve with phi_J
                        k2_J = max(J - k2 - k1 - oversampling, 0)

                        S_2_c = cdgmm(U_2_hat, phi[k1 + k2])
                        S_2_hat = subsample_fourier(S_2_c, 2**k2_J)
                        S_2_r = fft(S_2_hat, 'C2R', inverse=True)

                        S_2 = unpad(S_2_r, ind_start[k1 + k2 + k2_J], ind_end[k1 + k2 + k2_J])
                    else:
                        # just take the real value and unpad
                        U_2_r = real(U_2_m)
                        S_2 = unpad(U_2_r, ind_start[k1 + k2], ind_end[k1 + k2])

                    out_S_2.append({'coef': S_2,
                                    'j': (j1, j2),
                                    'n': (n1, n2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array' and vectorize:
        out_S = concatenate([x['coef'] for x in out_S])
    elif out_type == 'array' and not vectorize:
        out_S = {x['n']: x['coef'] for x in out_S}
    elif out_type == 'list':
        # NOTE: This overrides the vectorize flag.
        for x in out_S:
            x.pop('n')

    return out_S

__all__ = ['scattering1d']
