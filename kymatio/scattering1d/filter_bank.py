import numpy as np
import math
import warnings
from scipy.fft import ifft

def adaptive_choice_P(sigma, eps=1e-7):
    """
    Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the Fourier transform of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of
    * a Gabor term hat psi(omega) = hat g_{sigma}(omega - xi)
    where 0 < xi < 1 is some frequency and g_{sigma} is
    the Gaussian window defined in Fourier by
    hat g_{sigma}(omega) = e^{-omega^2/(2 sigma^2)}
    * a low pass term \\hat \\phi which is proportional to \\hat g_{\\sigma}.

    If \\sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the Fourier transform of both filters on the interval [1-P, P], the
    magnitude of the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

    |\\hat \\psi(P)| <= eps and |\\hat \\phi(1-P)| <= eps

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\\sigma} in Fourier.

    Parameters
    ----------
    sigma: float
        Positive number controlling the bandwidth of the filters
    eps : float, optional
        Positive number containing required precision. Defaults to 1e-7

    Returns
    -------
    P : int
        integer controlling the number of periods used to ensure the
        periodicity of the final Morlet filter in the frequency interval
        [0, 1[. The value of P will lead to the use of the frequency
        interval [1-P, P[, so that there are 2*P - 1 periods.
    """
    val = math.sqrt(-2 * (sigma**2) * math.log(eps))
    P = int(math.ceil(val + 1))
    return P


def periodize_filter_fourier(h_f, nperiods=1):
    """
    Computes a periodization of a filter provided in the Fourier domain.

    Parameters
    ----------
    h_f : array_like
        complex numpy array of shape (N*n_periods,)
    n_periods: int, optional
        Number of periods which should be used to periodize

    Returns
    -------
    v_f : array_like
        complex numpy array of size (N,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * N + k]
    """
    N = h_f.shape[0] // nperiods
    v_f = h_f.reshape(nperiods, N).mean(axis=0)
    return v_f


def morlet_1d(N, xi, sigma):
    """
    Computes the Fourier transform of a Morlet or Gauss filter.
    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - kappa)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and kappa is
    a corrective term which ensures that psi has a null average.
    If xi is None, the definition becomes: phi(t) = g_{sigma}(t)
    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float or None
        center frequency in (0, 1]
    sigma : float
        bandwidth parameter
    Returns
    -------
    filter_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    # Find the adequate value of P<=5
    P = min(adaptive_choice_P(sigma), 5)
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    if xi:
        # define the gabor at freq xi and the low-pass, both of width sigma
        gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
        # discretize in signal <=> periodize in Fourier
        gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
        # find the summation factor to ensure that morlet_f[0] = 0.
        kappa = gabor_f[0] / low_pass_f[0]
        filter_f = gabor_f - kappa * low_pass_f # psi (band-pass) case
    else:
        filter_f = low_pass_f # phi (low-pass) case
    filter_f /= np.abs(ifft(filter_f)).sum()
    return filter_f


def gauss_1d(N, sigma):
    return morlet_1d(N, xi=None, sigma=sigma)


def compute_sigma_psi(xi, Q, r=math.sqrt(0.5)):
    """
    Computes the frequential width sigma for a Morlet filter of frequency xi
    belonging to a family with Q wavelets.

    The frequential width is adapted so that the intersection of the
    frequency responses of the next filter occurs at a r-bandwidth specified
    by r, to ensure a correct coverage of the whole frequency axis.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    Q : int
        number of filters per octave, Q is an integer >= 1
    r : float, optional
        Positive parameter defining the bandwidth to use.
        Should be < 1. We recommend keeping the default value.
        The larger r, the larger the filters in frequency domain.

    Returns
    -------
    sigma : float
        frequential width of the Morlet wavelet.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    factor = 1. / math.pow(2, 1. / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / math.sqrt(2 * math.log(1. / r))
    return xi * term1 * term2


def compute_temporal_support(h_f, criterion_amplitude=1e-3):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain

    This function computes the support N which is the smallest integer
    such that for all signals x and all filters h,

    \\| x \\conv h - x \\conv h_{[-N, N]} \\|_{\\infty} \\leq \\epsilon
        \\| x \\|_{\\infty}  (1)

    where 0<\\epsilon<1 is an acceptable error, and h_{[-N, N]} denotes the
    filter h whose support is restricted in the interval [-N, N]

    The resulting value N used to pad the signals to avoid boundary effects
    and numerical errors.

    If the support is too small, no such N might exist.
    In this case, N is defined as the half of the support of h, and a
    UserWarning is raised.

    Parameters
    ----------
    h_f : array_like
        a numpy array of size batch x time, where each row contains the
        Fourier transform of a filter which is centered and whose absolute
        value is symmetric
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_f

    """
    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    # compute ||h - h_[-N, N]||_1
    l1_residual = np.fliplr(
        np.cumsum(np.fliplr(np.abs(h)[:, :half_support]), axis=1))
    # find the first point above criterion_amplitude
    if np.any(np.max(l1_residual, axis=0) <= criterion_amplitude):
        # if it is possible
        N = np.min(
            np.where(np.max(l1_residual, axis=0) <= criterion_amplitude)[0])\
            + 1
    else:
        # if there is none:
        N = half_support
        # Raise a warning to say that there will be border effects
        warnings.warn('Signal support is too small to avoid border effects')
    return N


def get_max_dyadic_subsampling(xi, sigma, alpha):
    """
    Computes the maximal dyadic subsampling which is possible for a Gabor
    filter of frequency xi and width sigma

    Finds the maximal integer j such that:
    omega_0 < 2^{-(j + 1)}
    where omega_0 is the boundary of the filter, defined as
    omega_0 = xi + alpha * sigma

    This ensures that the filter can be subsampled by a factor 2^j without
    aliasing.

    We use the same formula for Gabor and Morlet filters.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    sigma : float
        frequential width of the filter
    alpha : float, optional
        parameter controlling the error done in the aliasing.
        The larger alpha, the smaller the error.

    Returns
    -------
    j : int
        integer such that 2^j is the maximal subsampling accepted by the
        Gabor filter without aliasing.
    """
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j


def compute_xi_max(Q):
    """
    Computes the maximal xi to use for the Morlet family, depending on Q.

    Parameters
    ----------
    Q : int
        number of wavelets per octave (integer >= 1)

    Returns
    -------
    xi_max : float
        largest frequency of the wavelet frame.
    """
    xi_max = max(1. / (1. + math.pow(2., 3. / Q)), 0.35)
    return xi_max


def compute_params_filterbank(sigma_min, Q, alpha, r_psi=math.sqrt(0.5)):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    width of adjacent filters, up to a minimum frequency where the frequencies
    are translated. sigma_min limits the smallest frequential width
    among all filters, while preserving the coverage of the whole frequency
    axis.
    Parameters
    ----------
    sigma_min : float
        This acts as a lower bound on the frequential widths of the band-pass
        filters. The low-pass filter may be wider (if T < _N_padded), making
        invariants over shorter time scales than longest band-pass filter.
    Q : int
        number of wavelets per octave.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    Returns
    -------
    xis : list
        central frequencies of the filters.
    sigmas : list
        bandwidths of the filters.
    js : list
        maximal dyadic subsampling accepted by the filters.
        j=0 stands for no subsampling, j=1 stands for half subsampling, etc.
    """
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    if sigma_max <= sigma_min:
        xis = []
        sigmas = []
        elbow_xi = sigma_max
    else:
        xis =  [xi_max]
        sigmas = [sigma_max]

        # High-frequency (constant-Q) region: geometric progression of xi
        while sigmas[-1] > (sigma_min * math.pow(2, 1/Q)):
            xis.append(xis[-1] / math.pow(2, 1/Q))
            sigmas.append(sigmas[-1] / math.pow(2, 1/Q))
        elbow_xi = xis[-1]

    # Low-frequency (constant-bandwidth) region: arithmetic progression of xi
    for q in range(1, Q):
        xis.append(elbow_xi - q/Q * elbow_xi)
        sigmas.append(sigma_min)

    js = [
        get_max_dyadic_subsampling(xi, sigma, alpha) for xi, sigma in zip(xis, sigmas)
    ]
    return xis, sigmas, js


def scattering_filter_factory(N, J, Q, T, r_psi=math.sqrt(0.5),
                              max_subsampling=None, sigma0=0.1, alpha=5., **kwargs):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * k where k is an integer bounded below by 0. The maximal value for k
        depends on the type of filter, it is dynamically chosen depending
        on max_subsampling and the characteristics of the filters.
        Each value for k is an array (or tensor) of size 2**(J_support - k)
        containing the Fourier transform of the filter after subsampling by
        2**k

    Parameters
    ----------
    N : int
        padded length of the input signal. Corresponds to self._N_padded for the
        scattering object.
    J : int
        log-scale of the scattering transform, such that wavelets of both
        filterbanks have a maximal support that is proportional to 2**J.
    Q : tuple
        number of wavelets per octave at the first and second order 
        Q = (Q1, Q2). Q1 and Q2 are both int >= 1.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    max_subsampling: int or None, optional
        maximal dyadic subsampling to compute, in order
        to save computation time if it is not required. Defaults to None, in
        which case this value is dynamically adjusted depending on the filters.
    sigma0 : float, optional
        parameter controlling the frequential width of the low-pass filter at
        j=0; at a an absolute J, it is equal to sigma0 / 2**J. Defaults to 0.1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.

    Returns
    -------
    phi_f : dictionary
        a dictionary containing the low-pass filter at all possible
        subsamplings. See above for a description of the dictionary structure.
        The possible subsamplings are controlled by the inputs they can
        receive, which correspond to the subsamplings performed on top of the
        1st and 2nd order transforms.
    psi1_f : dictionary
        a dictionary containing the band-pass filters of the 1st order,
        only for the base resolution as no subsampling is used in the
        scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of the type (j, n) where n is an
        integer counting the filters and j the maximal dyadic subsampling
        which can be performed on top of the filter without aliasing.
    psi2_f : dictionary
        a dictionary containing the band-pass filters of the 2nd order
        at all possible subsamplings. The subsamplings are determined by the
        input they can receive, which depends on the scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of th etype (j, n) where n is an
        integer counting the filters and j is the maximal dyadic subsampling
        which can be performed on top of this filter without aliasing.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    # compute the spectral parameters of the filters
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s, j1s = compute_params_filterbank(sigma_min, Q1, alpha, r_psi)
    xi2s, sigma2s, j2s = compute_params_filterbank(sigma_min, Q2, alpha, r_psi)

    # width of the low-pass filter
    sigma_low = sigma0 / T

    # instantiate the dictionaries which will contain the filters
    phi_f = {}
    psi1_f = []
    psi2_f = []

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    for (xi2, sigma2, j2) in zip(xi2s, sigma2s, j2s):
        # compute the current value for the max_subsampling,
        # which depends on the input it can accept.
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [j1 for j1 in j1s if j2 > j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling
        # We first compute the filter without subsampling

        psi_levels = [morlet_1d(N, xi2, sigma2)]
        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for level in range(1, max_sub_psi2 + 1):
            nperiods = 2**level
            psi_levels.append(periodize_filter_fourier(psi_levels[0], nperiods))
        psi2_f.append({'levels': psi_levels, 'xi': xi2, 'sigma': sigma2, 'j': j2})

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with N=2**J_support
    for (xi1, sigma1, j1) in zip(xi1s, sigma1s, j1s):
        psi_levels = [morlet_1d(N, xi1, sigma1)]
        psi1_f.append({'levels': psi_levels, 'xi': xi1, 'sigma': sigma1, 'j': j1})

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    log2_T = math.floor(math.log2(T))
    if max_subsampling is None:
        max_subsampling_after_psi1 = max(j1s)
        max_subsampling_after_psi2 = max(j2s)
        max_sub_phi = min(max(max_subsampling_after_psi1,
                              max_subsampling_after_psi2), log2_T)
    else:
        max_sub_phi = max_subsampling

    # compute the filters at all possible subsamplings
    phi_levels = [gauss_1d(N, sigma_low)]
    for level in range(1, max_sub_phi + 1):
        nperiods = 2**level
        phi_levels.append(periodize_filter_fourier(phi_levels[0], nperiods))
    phi_f = {'levels': phi_levels, 'xi': 0, 'sigma': sigma_low, 'j': log2_T}

    # return results
    return phi_f, psi1_f, psi2_f
