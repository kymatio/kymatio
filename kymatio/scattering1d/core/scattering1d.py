
def scattering1d(U_0, backend, psi1, psi2, phi, ind_start=None, ind_end=None,
        oversampling=0, max_order=2, average=True):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    U_0 : Tensor
        a torch Tensor of size `(B, 1, N)` where `N` is the temporal size
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
    max_order : int, optional
        Number of orders in the scattering transform. Either 1 or 2 (default).
    average : boolean, optional
        whether to average the first order vector. Defaults to `True`
    """
    cdgmm = backend.cdgmm
    ifft = backend.ifft
    irfft = backend.irfft
    modulus = backend.modulus
    rfft = backend.rfft
    subsample_fourier = backend.subsample_fourier
    unpad = backend.unpad

    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    # Get S0
    log2_T = phi["j"]
    k0 = max(log2_T - oversampling, 0)

    if average:
        S_0_c = cdgmm(U_0_hat, phi[0])
        S_0_hat = subsample_fourier(S_0_c, 2**k0)
        S_0_r = irfft(S_0_hat)

        coef = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        coef = unpad(U_0, ind_start[0], ind_end[0])
    yield {'coef': coef, 'j': (), 'n': ()}

    # First order:
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']

        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)

        assert psi1[n1]['xi'] < 0.5 / (2**k1)
        U_1_c = cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)

        # Take the modulus
        U_1_m = modulus(U_1_c)

        if average or max_order > 1:
            U_1_hat = rfft(U_1_m)

        if average:
            # Convolve with phi_J
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi[k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = irfft(S_1_hat)

            coef = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            coef = unpad(U_1_m, ind_start[k1], ind_end[k1])

        yield {'coef': coef, 'j': (j1,), 'n': (n1,)}

        if max_order == 2:
            # 2nd order
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']

                if j2 > j1:
                    assert psi2[n2]['xi'] < psi1[n1]['xi']

                    # convolution + downsampling
                    sub2_adj = min(j2, log2_T) if average else j2
                    k2 = max(sub2_adj - k1 - oversampling, 0)

                    U_2_c = cdgmm(U_1_hat, psi2[n2][k1])
                    U_2_hat = subsample_fourier(U_2_c, 2**k2)
                    # take the modulus
                    U_2_c = ifft(U_2_hat)

                    U_2_m = modulus(U_2_c)

                    if average:
                        U_2_hat = rfft(U_2_m)

                        # Convolve with phi_J
                        k2_log2_T = max(log2_T - k2 - k1 - oversampling, 0)

                        S_2_c = cdgmm(U_2_hat, phi[k1 + k2])
                        S_2_hat = subsample_fourier(S_2_c, 2**k2_log2_T)
                        S_2_r = irfft(S_2_hat)

                        coef = unpad(S_2_r, ind_start[k1 + k2 + k2_log2_T],
                                    ind_end[k1 + k2 + k2_log2_T])
                    else:
                        coef = unpad(U_2_m, ind_start[k1 + k2], ind_end[k1 + k2])

                    yield {'coef': coef, 'j': (j1, j2), 'n': (n1, n2)}

__all__ = ['scattering1d']
