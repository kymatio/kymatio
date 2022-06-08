from ..utils import compute_border_indices, compute_padding
from operator import itemgetter

def scattering1d(x, backend, psi1, psi2, phi, oversampling=0, max_order=2,
    average=True, vectorize=False, out_type='array'):
    """
    Backend-agnostic implementation of the 1-D scattering transform.

    Parameters
    ----------
    x : array
        input signal
    backend : Python module
        as returned by self._instantiate_backend
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
    oversampling : int, optional
        how much to oversample the scattering (with respect to :math:`2^J`):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to `0`
    max_order : int, optional
        Order of the scattering transform. Either `1` or `2` (default).
    average : boolean, optional
        whether to average the scattering representation. Defaults to `True`
    vectorize : boolean, optional
        whether to return a dictionary or a tensor. Defaults to False.
    """
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    pad = backend.pad
    unpad = backend.unpad

    # Prepare for padding/unpadding
    N = x.shape[-1]
    N_pad = len(phi[0])
    pad_left, pad_right = compute_padding(N, N_pad)
    log2_T = phi["j"]
    J = max(max(map(itemgetter("j"), psi1)), max(map(itemgetter("j"), psi2)))
    ind_start, ind_end = compute_border_indices(log2_T, J, pad_left, pad_left+N)

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)
    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    # Zeroth order
    k0 = max(log2_T - oversampling, 0)

    if average:
        S_0_c = cdgmm(U_0_hat, phi[0])
        S_0_hat = subsample_fourier(S_0_c, 2**k0)
        S_0_r = irfft(S_0_hat)

        S_0 = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x
    out_S = [{'coef': S_0,
              'j': (),
              'n': ()}]

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

            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])

        out_S.append({'coef': S_1,
                      'j': (j1,),
                      'n': (n1,)})

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

                        S_2 = unpad(S_2_r, ind_start[k1 + k2 + k2_log2_T],
                                    ind_end[k1 + k2 + k2_log2_T])
                    else:
                        S_2 = unpad(U_2_m, ind_start[k1 + k2], ind_end[k1 + k2])

                    out_S.append({'coef': S_2,
                                  'j': (j1, j2),
                                  'n': (n1, n2)})

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
