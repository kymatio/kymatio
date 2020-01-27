import scipy.fftpack
import warnings

def compute_padding(M, N, J):
    """
         Precomputes the future padded size. If 2^J=M or 2^J=N,
         border effects are unavoidable in this case, and it is
         likely that the input has either a compact support,
         either is periodic.

         Parameters
         ----------
         M, N : int
             input size

         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J

    return M_padded, N_padded

def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)


def compute_meta_scattering(J, L, max_order=2):
    """Get metadata on the transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    L : int >= 1
        The number of angles used.
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.

    Returns
    -------
    meta_phi, meta_psi: filters with their des cription
        Each filter is filled with the following keys:

        - `'theta'` : int
            The corresponding rotation angle related to a filter.
        - `'j'` : int
            The dyadic scale of the filter used at each order.
    """


    meta_phi = {'j': J}

    meta_psi = {}


    for j1 in range(J):
        for theta1 in range(L):
            f1 = {}
            f1['theta'] = theta1
            f1['j'] = j1
            meta_psi[(j1,)] = f1

            if max_order < 2:
                continue

            for j2 in range(J):
                for theta2 in range(L):
                    if j2 > j1:
                        f2 = {}
                        f2['theta'] = theta2
                        f2['j'] = j2
                        meta_psi[(j1, j2)] = (f1, f2)

    return meta_phi, meta_psi