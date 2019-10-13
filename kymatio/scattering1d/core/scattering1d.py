# Authors: Mathieu Andreux, Joakim Anden, Edouard Oyallon
# Scientific Ancestry: Joakim Anden, Mathieu Andreux, Vincent Lostanlen

__all__ = ['scattering1d']

def scattering1d(x, backend, J, psi1, psi2, phi, pad_left=0, pad_right=0,
               ind_start=None, ind_end=None, oversampling=0,
               max_order=2, average=True, size_scattering=(0, 0, 0), vectorize=False):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    x : Tensor | Numpy array
        A tensor/numpy array of size `(B, 1, T)` where `T` is the temporal size.
    backend : named tuple
        Named tuple which holds all functions needed for the scattering
        transform.
    psi1 : dictionary
        A dictionary of filters (in the Fourier domain), with keys (`j`, `q`).
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
        A dictionary of filters, with keys (j2, n2). Same remarks as for psi1
    phi : dictionary
        A dictionary of filters of scale :math:`2^J` with keys (`j`)
        where :math:`2^j` is the downsampling factor.
        The array `phi[j]` is a real-valued filter.
    J : int
        Scale of the scattering.
    pad_left : int, optional
        How much to pad the signal on the left. Defaults to `0`.
    pad_right : int, optional
        How much to pad the signal on the right. Defaults to `0`.
    ind_start : dictionary of ints, optional
        Indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None.
    ind_end : dictionary of ints, optional
        See description of ind_start.
    oversampling : int, optional
        How much to oversample the scattering (with respect to :math:`2^J`):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to `0`.
    order2 : boolean, optional
        Whether to compute the 2nd order or not. Defaults to `False`.
    average_U1 : boolean, optional
        Whether to average the first order vector. Defaults to `True`.
    size_scattering : tuple
        Contains the number of channels of the scattering, precomputed for
        speed-up. Defaults to `(0, 0, 0)`.
    vectorize : boolean, optional
        Whether to return a dictionary or a tensor. Defaults to False.

    """
    subsample_fourier, modulus_complex, fft1d_c2c, ifft1d_c2c, real, pad,\
    unpad, finalize = backend.subsample_fourier, backend.modulus_complex, backend.fft1d_c2c,\
    backend.ifft1d_c2c,  backend.real, backend.pad, backend.unpad, backend.finalize

    # S is simply a dictionary if we do not perform the averaging...
    if vectorize:
        batch_size = x.shape[0]
        kJ = max(J - oversampling, 0)
        temporal_size = ind_end[kJ] - ind_start[kJ]
        out_S_0, out_S_1, out_S_2 = [], [], []
    else:
        S = {}

    # pad to a dyadic size and make it complex
    U0 = pad(x, pad_left=pad_left, pad_right=pad_right, to_complex=True)
    # compute the Fourier transform
    U0_hat = fft1d_c2c(U0)
    if vectorize:
        # initialize the cursor
        cc = [0] + list(size_scattering[:-1])  # current coordinate
        cc[1] = cc[0] + cc[1]
        if max_order == 2:
            cc[2] = cc[1] + cc[2]
    # Get S0
    k0 = max(J - oversampling, 0)
    if average:
        S0_J_hat = subsample_fourier(U0_hat * phi[0], 2**k0)
        S0_J = unpad(real(ifft1d_c2c(S0_J_hat)),
                     ind_start[k0], ind_end[k0])
    else:
        S0_J = x
    if vectorize:
        out_S_0.append(S0_J)
        cc[0] += 1
    else:
        S[()] = S0_J
    # First order:
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)
        assert psi1[n1]['xi'] < 0.5 / (2**k1)
        U1_hat = subsample_fourier(U0_hat * psi1[n1][0], 2**k1)
        # Take the modulus
        U1 = modulus_complex(ifft1d_c2c(U1_hat))
        if average or max_order > 1:
            U1_hat = fft1d_c2c(U1)
        if average:
            # Convolve with phi_J
            k1_J = max(J - k1 - oversampling, 0)
            S1_J_hat = subsample_fourier(U1_hat * phi[k1], 2**k1_J)
            S1_J = unpad(real(ifft1d_c2c(S1_J_hat)),
                         ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            # just take the real value and unpad
            S1_J = unpad(real(U1), ind_start[k1], ind_end[k1])
        if vectorize:
            out_S_1.append(S1_J)
            cc[1] += 1
        else:
            S[(n1,)] = S1_J
        if max_order == 2:
            # 2nd order
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']
                if j2 > j1:
                    assert psi2[n2]['xi'] < psi1[n1]['xi']
                    # convolution + downsampling
                    k2 = max(j2 - k1 - oversampling, 0)
                    U2_hat = subsample_fourier(U1_hat * psi2[n2][k1],
                                               2**k2)
                    # take the modulus and go back in Fourier
                    U2 = modulus_complex(ifft1d_c2c(U2_hat))
                    if average:
                        U2_hat = fft1d_c2c(U2)
                        # Convolve with phi_J
                        k2_J = max(J - k2 - k1 - oversampling, 0)
                        S2_J_hat = subsample_fourier(U2_hat * phi[k1 + k2],
                                                     2**k2_J)
                        S2_J = unpad(real(ifft1d_c2c(S2_J_hat)),
                                     ind_start[k1 + k2 + k2_J],
                                     ind_end[k1 + k2 + k2_J])
                    else:
                        # just take the real value and unpad
                        S2_J = unpad(
                            real(U2), ind_start[k1 + k2], ind_end[k1 + k2])
                    if vectorize:
                        out_S_2.append(S2_J)
                        cc[2] += 1
                    else:
                        S[n1, n2] = S2_J

    if vectorize:
        S = finalize(out_S_0, out_S_1, out_S_2)
    return S

