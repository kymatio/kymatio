
def scattering1d(U_0, backend, filters, log2_stride, average_local):
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
    log2_stride : int >=0
        Yields coefficients with a temporal stride equal to 2**log2_stride.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.
    """

    # compute the Fourier transform
    U_0_hat = backend.rfft(U_0)

    # Get S0
    phi = filters[0]
    log2_T = phi['j']
    stride = 2**log2_stride

    if average_local:
        S_0_c = backend.cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = backend.subsample_fourier(S_0_c, stride)
        S_0_r = backend.irfft(S_0_hat)
        yield {'coef': S_0_r, 'j': (), 'n': ()}
    else:
        yield {'coef': U_0, 'j': (), 'n': ()}

    # First order:
    psi1 = filters[1]
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = min(j1, log2_stride) if average_local else j1
        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2**k1)
        U_1_c = backend.ifft(U_1_hat)

        # Take the modulus
        U_1_m = backend.modulus(U_1_c)

        if average_local or (len(filters) > 2):
            U_1_hat = backend.rfft(U_1_m)

        if average_local:
            # Convolve with phi_J
            k1_J = max(log2_stride - k1, 0)
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
                    sub2_adj = min(j2, log2_stride) if average_local else j2
                    k2 = max(sub2_adj - k1, 0)
                    U_2_c = backend.cdgmm(U_1_hat, psi2[n2]['levels'][k1])
                    U_2_hat = backend.subsample_fourier(U_2_c, 2**k2)
                    U_2_c = backend.ifft(U_2_hat)

                    # take the modulus
                    U_2_m = backend.modulus(U_2_c)

                    if average_local:
                        # Convolve with phi_J
                        U_2_hat = backend.rfft(U_2_m)
                        k2_log2_T = max(log2_stride - sub2_adj, 0)
                        S_2_c = backend.cdgmm(U_2_hat, phi['levels'][k1+k2])
                        S_2_hat = backend.subsample_fourier(S_2_c, 2**k2_log2_T)
                        S_2_r = backend.irfft(S_2_hat)
                        yield {'coef': S_2_r, 'j': (j1, j2), 'n': (n1, n2)}
                    else:
                        yield {'coef': U_2_m, 'j': (j1, j2), 'n': (n1, n2)}

__all__ = ['scattering1d']
