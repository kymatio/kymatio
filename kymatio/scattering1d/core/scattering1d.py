
def scattering1d(U_0, backend, filters, oversampling, average=True):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    U_0 : Tensor
        an backend-compatible array of size `(B, 1, N)` where `B` is batch size
        and `N` is the padded signal length.
    backend : module
        Kymatio module which matches the type of U_0.
    psi1 : list
        a list of dictionaries, expressing wavelet band-pass filters in the
        Fourier domain for the first layer of the scattering transform.
        Each `psi1[n1]` is a dictionary with keys:
            * `j`: int, subsampling factor
            * `xi`: float, center frequency
            * `sigma`: float, bandwidth
            * `levels`: list, values taken by the wavelet in the Fourier domain
                        at different levels of detail.
        Each psi1[n]['levels'][level] is an array with size N/2**level.
    psi2 : dictionary
        Same as psi1, but for the second layer of the scattering transform.
    phi : dictionary
        a dictionary expressing the low-pass filter in the Fourier domain.
        Keys:
        * `j`: int, subsampling factor (also known as log_T)
        * `xi`: float, center frequency (=0 by convention)
        * `sigma`: float, bandwidth
        * 'levels': list, values taken by the lowpass in the Fourier domain
                    at different levels of detail.
    oversampling : int, optional
        how much to oversample the scattering (with respect to `log2_T`):
        the higher, the larger the resulting scattering
        tensor along time. Must be nonnegative (`oversampling>=0     ).
    max_order : int, optional
        Number of orders in the scattering transform. Either 1 or 2.
    average : boolean, optional
        whether to average the result. Default to True.
    """

    # compute the Fourier transform
    U_0_hat = backend.rfft(U_0)

    # Get S0
    phi = filters[0]
    log2_T = phi['j']
    k0 = max(log2_T - oversampling, 0)

    if average:
        S_0_c = backend.cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = backend.subsample_fourier(S_0_c, 2**k0)
        S_0_r = backend.irfft(S_0_hat)
        yield {'coef': S_0_r, 'j': (), 'n': ()}
    else:
        yield {'coef': U_0, 'j': (), 'n': ()}

    # First order:
    psi1 = filters[1]
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']

        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)

        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2**k1)
        U_1_c = backend.ifft(U_1_hat)

        # Take the modulus
        U_1_m = backend.modulus(U_1_c)

        if average or len(filters) > 2:
            U_1_hat = backend.rfft(U_1_m)

        if average:
            # Convolve with phi_J
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_c = backend.cdgmm(U_1_hat, phi['levels'][k1])
            S_1_hat = backend.subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = backend.irfft(S_1_hat)
            yield {'coef': S_1_r, 'j': (j1,), 'n': (n1,)}
        else:
            yield {'coef': U_1_m, 'j': (j1,), 'n': (n1,)}

        if len(filters) > 2:
            # 2nd order
            psi2 = filters[2]
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']

                if j2 > j1:
                    # convolution + downsampling
                    sub2_adj = min(j2, log2_T) if average else j2
                    k2 = max(sub2_adj - k1 - oversampling, 0)

                    U_2_c = backend.cdgmm(U_1_hat, psi2[n2]['levels'][k1])
                    U_2_hat = backend.subsample_fourier(U_2_c, 2**k2)
                    # take the modulus
                    U_2_c = backend.ifft(U_2_hat)

                    U_2_m = backend.modulus(U_2_c)

                    if average:
                        U_2_hat = backend.rfft(U_2_m)

                        # Convolve with phi_J
                        k2_log2_T = max(log2_T - k2 - k1 - oversampling, 0)

                        S_2_c = backend.cdgmm(U_2_hat, phi['levels'][k1 + k2])
                        S_2_hat = backend.subsample_fourier(S_2_c, 2**k2_log2_T)
                        S_2_r = backend.irfft(S_2_hat)

                        yield {'coef': S_2_r, 'j': (j1, j2), 'n': (n1, n2)}
                    else:
                        yield {'coef': U_2_m, 'j': (j1, j2), 'n': (n1, n2)}

__all__ = ['scattering1d']
