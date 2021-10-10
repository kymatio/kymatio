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


def morlet_1d(N, xi, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the Fourier transform of a Morlet filter.

    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        central frequency (in [0, 1])
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'.
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max: int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)

    Returns
    -------
    morlet_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    # Find the adequate value of P
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    # define the gabor at freq xi and the low-pass, both of width sigma
    gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))
    # discretize in signal <=> periodize in Fourier
    gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    # find the summation factor to ensure that morlet_f[0] = 0.
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    # normalize the Morlet if necessary
    morlet_f *= get_normalizing_factor(morlet_f, normalize=normalize)
    return morlet_f


def get_normalizing_factor(h_f, normalize='l1'):
    """
    Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_f : array_like
        numpy vector containing the Fourier transform of a filter
    normalized : string, optional
        desired normalization type, either 'l1' or 'l2'. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that h_f * norm_factor is the adequately normalized vector.
    """
    h_real = ifft(h_f)
    if np.abs(h_real).sum() < 1e-7:
        raise ValueError('Zero division error is very likely to occur, ' +
                         'aborting computations now.')
    if normalize == 'l1':
        norm_factor = 1. / (np.abs(h_real).sum())
    elif normalize == 'l2':
        norm_factor = 1. / np.sqrt((np.abs(h_real)**2).sum())
    else:
        raise ValueError("Supported normalizations only include 'l1' and 'l2'")
    return norm_factor


def gauss_1d(N, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the Fourier transform of a low pass gaussian window.

    \\hat g_{\\sigma}(\\omega) = e^{-\\omega^2 / 2 \\sigma^2}

    Parameters
    ----------
    N : int
        size of the temporal support
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max : int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float, optional
        required machine precision (to choose the adequate P)

    Returns
    -------
    g_f : array_like
        numpy array of size (N,) containing the Fourier transform of the
        filter (with the frequencies in the np.fft.fftfreq convention).
    """
    # Find the adequate value of P
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # switch cases
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    # define the low pass
    g_f = np.exp(-freqs_low**2 / (2 * sigma**2))
    # periodize it
    g_f = periodize_filter_fourier(g_f, nperiods=2 * P - 1)
    # normalize the signal
    g_f *= get_normalizing_factor(g_f, normalize=normalize)
    # return the Fourier transform
    return g_f


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


def get_max_dyadic_subsampling(xi, sigma, alpha=5.):
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
        The larger alpha, the smaller the error. Defaults to 5.

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


def move_one_dyadic_step(cv, Q, alpha=5.):
    """
    Computes the parameters of the next wavelet on the low frequency side,
    based on the parameters of the current wavelet.

    This function is used in the loop defining all the filters, starting
    at the wavelet frequency and then going to the low frequencies by
    dyadic steps. This makes the loop in compute_params_filterbank much
    simpler to read.

    The steps are defined as:
    xi_{n+1} = 2^{-1/Q} xi_n
    sigma_{n+1} = 2^{-1/Q} sigma_n

    Parameters
    ----------
    cv : dictionary
        stands for current_value. Is a dictionary with keys:
        *'key': a tuple (j, n) where n is a counter and j is the maximal
            dyadic subsampling accepted by this wavelet.
        *'xi': central frequency of the wavelet
        *'sigma': width of the wavelet
    Q : int
        number of wavelets per octave. Controls the relationship between
        the frequency and width of the current wavelet and the next wavelet.
    alpha : float, optional
        tolerance parameter for the aliasing. The larger alpha,
        the more conservative the algorithm is. Defaults to 5.

    Returns
    -------
    new_cv : dictionary
        a dictionary with the same keys as the ones listed for cv,
        whose values are updated
    """
    factor = 1. / math.pow(2., 1. / Q)
    n = cv['key']
    new_cv = {'xi': cv['xi'] * factor, 'sigma': cv['sigma'] * factor}
    # compute the new j
    new_cv['j'] = get_max_dyadic_subsampling(new_cv['xi'], new_cv['sigma'], alpha=alpha)
    new_cv['key'] = n + 1
    return new_cv


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


def compute_params_filterbank(sigma_low, Q, r_psi=math.sqrt(0.5), alpha=5.):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    width of adjacent filters, up to a minimum frequency where the frequencies
    are translated.
    This ensures that the low-pass filter has the largest temporal support
    among all filters, while preserving the coverage of the whole frequency
    axis.

    The keys of the dictionaries are tuples of integers (j, n) where n is a
    counter (starting at 0 for the highest frequency filter) and j is the
    maximal dyadic subsampling accepted by this filter.

    Parameters
    ----------
    sigma_low : float
        frequential width of the low-pass filter. This acts as a
        lower-bound on the frequential widths of the band-pass filters,
        so as to ensure that the low-pass filter has the largest temporal
        support among all filters.
    Q : int
        number of wavelets per octave.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.

    Returns
    -------
    xi : dictionary
        dictionary containing the central frequencies of the wavelets.
    sigma : dictionary
        dictionary containing the frequential widths of the wavelets.

    Refs
    ----
    Convolutional operators in the time-frequency domain, 2.1.3, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    xi = []
    sigma = []
    j = []

    if sigma_max <= sigma_low:
        # in this exceptional case, we will not go through the loop, so
        # we directly assign
        last_xi = sigma_max
    else:
        # fill all the dyadic wavelets as long as possible
        current = {'key': 0, 'j': 0, 'xi': xi_max, 'sigma': sigma_max}
        while current['sigma'] > sigma_low:  # while we can attribute something
            xi.append(current['xi'])
            sigma.append(current['sigma'])
            j.append(current['j'])
            current = move_one_dyadic_step(current, Q, alpha=alpha)
        # get the last key
        last_xi = xi[-1]
    # fill num_interm wavelets between last_xi and 0, both excluded
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_low
        xi.append(new_xi)
        sigma.append(new_sigma)
        j.append(get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha))
    # return results
    return xi, sigma, j


def calibrate_scattering_filters(J, Q, r_psi=math.sqrt(0.5), sigma0=0.1,
                                 alpha=5.):
    """
    Calibrates the parameters of the filters used at the 1st and 2nd orders
    of the scattering transform.

    These filterbanks share the same low-pass filterbank, but use a
    different Q: Q_1 = Q and Q_2 = 1.

    The dictionaries for the band-pass filters have keys which are 2-tuples
    of the type (j, n), where n is an integer >=0 counting the filters (for
    identification purposes) and j is an integer >= 0 denoting the maximal
    subsampling 2**j which can be performed on a signal convolved with this
    filter without aliasing.

    Parameters
    ----------
    J : int
        maximal scale of the scattering (controls the number of wavelets)
    Q : int
        number of wavelets per octave for the first order
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5)
    sigma0 : float, optional
        frequential width of the low-pass filter at scale J=0
        (the subsequent widths are defined by sigma_J = sigma0 / 2^J).
        Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.

    Returns
    -------
    sigma_low : float
        frequential width of the low-pass filter
    xi1 : dictionary
        dictionary containing the center frequencies of the first order
        filters. See above for a decsription of the keys.
    sigma1 : dictionary
        dictionary containing the frequential width of the first order
        filters. See above for a description of the keys.
    xi2 : dictionary
        dictionary containing the center frequencies of the second order
        filters. See above for a decsription of the keys.
    sigma2 : dictionary
        dictionary containing the frequential width of the second order
        filters. See above for a description of the keys.
    """
    if Q < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))
    sigma_low = sigma0 / math.pow(2, J)  # width of the low pass
    xi1, sigma1, j1 = compute_params_filterbank(sigma_low, Q, r_psi=r_psi,
                                            alpha=alpha)
    xi2, sigma2, j2 = compute_params_filterbank(sigma_low, 1, r_psi=r_psi,
                                            alpha=alpha)
    return sigma_low, xi1, sigma1, j1, xi2, sigma2, j2


def scattering_filter_factory(J_support, J_scattering, Q, r_psi=math.sqrt(0.5),
                              criterion_amplitude=1e-3, normalize='l1',
                              max_subsampling=None, sigma0=0.1, alpha=5.,
                              P_max=5, eps=1e-7, **kwargs):
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
    * 'width': temporal width

    Parameters
    ----------
    J_support : int
        2**J_support is the desired support size of the filters
    J_scattering : int
        parameter for the scattering transform (2**J_scattering
        corresponds to the averaging support of the low-pass filter)
    Q : int
        number of wavelets per octave at the first order. For audio signals,
        a value Q >= 12 is recommended in order to separate partials.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    criterion_amplitude : float, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding. Defaults to 1e-3.
    normalize : string, optional
        Normalization convention for the filters (in the
        temporal domain). Supported values include 'l1' and 'l2'; a ValueError
        is raised otherwise. Defaults to 'l1'.
    max_subsampling: int or None, optional
        maximal dyadic subsampling to compute, in order
        to save computation time if it is not required. Defaults to None, in
        which case this value is dynamically adjusted depending on the filters.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering, it
        is equal to sigma0 / 2**J_scattering. Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic. P_max = 5 is more than enough for
        double precision. Defaults to 5. Should be >= 1
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to 1e-7

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
    t_max_phi : int
        temporal size to use to pad the signal on the right and on the
        left by making at most criterion_amplitude error. Assumes that the
        temporal support of the low-pass filter is larger than all filters.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    # compute the spectral parameters of the filters
    sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = calibrate_scattering_filters(
        J_scattering, Q, r_psi=r_psi, sigma0=sigma0, alpha=alpha)

    # instantiate the dictionaries which will contain the filters
    phi_f = {}
    psi1_f = []
    psi2_f = []

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    for (n2, j2) in enumerate(j2s):
        # compute the current value for the max_subsampling,
        # which depends on the input it can accept.
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [
                j1 for j1 in j1s if j2 > j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling
        # We first compute the filter without subsampling
        N = 2**J_support

        psi_f = {}
        psi_f[0] = morlet_1d(
            N, xi2[n2], sigma2[n2], normalize=normalize, P_max=P_max,
            eps=eps)
        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for subsampling in range(1, max_sub_psi2 + 1):
            factor_subsampling = 2**subsampling
            psi_f[subsampling] = periodize_filter_fourier(
                psi_f[0], nperiods=factor_subsampling)
        psi2_f.append(psi_f)

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with N=2**J_support
    for (n1, j1) in enumerate(j1s):
        N = 2**J_support
        psi1_f.append({0: morlet_1d(
            N, xi1[n1], sigma1[n1], normalize=normalize,
            P_max=P_max, eps=eps)})

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    if max_subsampling is None:
        max_subsampling_after_psi1 = max(j1s)
        max_subsampling_after_psi2 = max(j2s)
        max_sub_phi = max(max_subsampling_after_psi1,
                          max_subsampling_after_psi2)
    else:
        max_sub_phi = max_subsampling

    # compute the filters at all possible subsamplings
    phi_f[0] = gauss_1d(N, sigma_low, P_max=P_max, eps=eps)
    for subsampling in range(1, max_sub_phi + 1):
        factor_subsampling = 2**subsampling
        # compute the low_pass filter
        phi_f[subsampling] = periodize_filter_fourier(
            phi_f[0], nperiods=factor_subsampling)

    # Embed the meta information within the filters
    for (n1, j1) in enumerate(j1s):
        psi1_f[n1]['xi'] = xi1[n1]
        psi1_f[n1]['sigma'] = sigma1[n1]
        psi1_f[n1]['j'] = j1
        psi1_f[n1]['width'] = {0: 2*compute_temporal_width(
            psi1_f[n1][0], sigma0=sigma0,
            criterion_amplitude=criterion_amplitude)}
    for (n2, j2) in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2[n2]
        psi2_f[n2]['sigma'] = sigma2[n2]
        psi2_f[n2]['j'] = j2
        for k in psi2_f[n2]:
            if isinstance(k, int):
                psi2_f[n2]['width'][k] = 2*compute_temporal_width(
                    psi2_f[n2][k], sigma0=sigma0,
                    criterion_amplitude=criterion_amplitude)
    phi_f['xi'] = 0.
    phi_f['sigma'] = sigma_low
    phi_f['j'] = 0
    phi_f['width'] = 2*compute_temporal_width(
        phi_f[0], sigma0=sigma0, criterion_amplitude=criterion_amplitude)

    # compute the support size allowing to pad without boundary errors
    # at the finest resolution
    t_max_phi = compute_temporal_support(
        phi_f[0].reshape(1, -1), criterion_amplitude=criterion_amplitude)

    # return results
    return phi_f, psi1_f, psi2_f, t_max_phi


def compute_temporal_width(p_f, N=None, pts_per_scale=6, fast=True,
                           sigma0=.1, criterion_amplitude=1e-3):
    """Measures "width" in terms of amount of invariance imposed via convolution.
    See below for detailed description.

    Parameters
    ----------
    p_f: np.ndarray
        Frequency-domain filter of length >= N. "Length" must be along dim0,
        i.e. `(freqs, ...)`.

    N: int / None
        Unpadded output length. (In scattering we convolve at e.g. x4 input's
        length, then unpad to match input's length).
        Defaults to `len(p_f) // 2`.

    pts_per_scale: int
        Used only in `fast=False`: number of Gaussians generated per dyadic
        scale. Greater improves approximation accuracy but is slower.

    sigma0, criterion_amplitude: float, float
        See `help(kymatio.scattering1d.filter_bank.gauss_1d)`. Parameters
        defining the Gaussian lowpass used as reference for computing `width`.
        That is, `width` is defined *in terms of* this Gaussian.

    Returns
    -------
    width: int
        The estimated width of `p_f`.

    Motivation
    ----------
    The measure is, a relative L2 distance (Euclidean distance relative to
    input norms) between inner products of `p_f` with an input, at different
    time shifts (i.e. `L2(A, B); A = sum(p(t) * x(t)), B = sum(p(t) * x(t - T))`).
      - The time shift is made to be "maximal", i.e. half of unpadded output
        length (`N/2`), which provides the most sensitive measure to "width".
      - The input is a Dirac delta, thus we're really measuring distances between
        impulse responses of `p_f`, or of `p_f` with itself. This provides a
        measure that's close to that of input being WGN, many-trials averaged.

    This yields `l2_reference`. It's then compared against `l2_Ts`, which is
    a list of measures obtained the same way except replacing `p_f` with a fully
    decayed (undistorted) Gaussian lowpass - and the `l2_T` which most closely
    matches `l2_reference` is taken to be the `width`.

    The interpretation is, "width" of `p_f` is the width of the (properly decayed)
    Gaussian lowpass that imposes the same amount of invariance that `p_f` does.

    Algorithm
    ---------
    The actual algorithm is different, since L2 is a problematic measure with
    unpadding; there's ambiguity: `T=1` can be measured as "closer" to `T=64`
    than `T=63`). Namely, we take inner product (similarity) of `p_f` with
    Gaussians at varied `T`, and the largest such product is the `width`.
    The result is very close to that described under "Motivation", but without
    the ambiguity that requires correction steps.

    Fast algorithm
    --------------
    Approximates `fast=False`. If `p_f` is fully decayed, the result is exactly
    same as `fast=False`. If `p_f` is very wide (nearly global average), the
    result is also same as `fast=False`. The disagreement is in the intermediate
    zone, but is not significant.

    We compute `ratio = p_t.max() / p_t.min()`, and compare against a fully
    decayed reference. For the intermediate zone, a quadratic interpolation
    is used to approximate `fast=False`.

    Assumption
    ----------
    `abs(p_t)`, where `p_t = ifft(p_f)`, is assumed to be Gaussian-like.
    An ideal measure is devoid of such an assumption, but is difficult to devise
    in finite sample settings.

    Note
    ----
    `width` saturates at `N` past a certain point in "incomplete decay" regime.
    The intution is, in incomplete decay, the measure of width is ambiguous:
    for one, we lack compact support. For other, the case `width == N` is a
    global averaging (simple mean, dc, etc), which is really `width = inf`.

    If we let the algorithm run with unrestricted `T_max`, we'll see `width`
    estimates blow up as `T -> N` - but we only seek to quantify invariance
    up to `N`. Also, Gaussians of widths `T = N - const` and `T = N` have
    very close L2 measures up to a generous `const`; see `test_global_averaging()`
    in `tests/scattering1d/test_jtfs.py` for `T = N - 1` (and try e.g. `N - 64`).
    """
    if len(p_f) == 1:  # edge case
        return 1

    # obtain temporal filter
    p_f = p_f.squeeze()
    p_t = np.abs(ifft(p_f))

    # relevant params
    Np = len(p_f)
    if N is None:
        N = Np // 2
    ca = dict(criterion_amplitude=criterion_amplitude)

    # compute "complete decay" factor
    uses_defaults = bool(sigma0 == .1 and criterion_amplitude == 1e-3)
    if uses_defaults:
        # precomputed
        complete_decay_factor = 16
        fast_approx_amp_ratio = 0.8208687174155399
    else:
        if fast:
            raise ValueError("`fast` requires using default values of "
                             "`sigma0` and `criterion_amplitude`.")
        T = Np
        phi_f_fn = lambda Np_phi: gauss_1d(Np_phi, sigma0 / T)
        Np_min = compute_minimum_required_length(phi_f_fn, Np, **ca)
        complete_decay_factor = 2 ** math.ceil(math.log2(Np_min / Np))
        phi_t = phi_f_fn(Np_min)
        fast_approx_amp_ratio = phi_t[T] / phi_t[0]

    if fast:
        ratio = (p_t / p_t[0])[:len(p_t)//2]  # assume ~symmetry about halflength
        rmin = ratio.min()
        if rmin > fast_approx_amp_ratio:
            # equivalent of `not complete_decay`
            th_global_avg = .96
            # never sufficiently decays
            if rmin > th_global_avg:
                return N
            # quadratic interpolation
            # y0 = A + B*x0^2
            # y1 = A + B*x1^2
            # B = (y0 - y1) / (x0^2 - x1^2)
            # A = y0 - B*x0^2
            y0 = .5 * Np
            y1 = N
            x0 = fast_approx_amp_ratio
            x1 = th_global_avg
            B = (y0 - y1) / (x0**2 - x1**2)
            A = y0 - B*x0**2
            T_est = A + B*rmin**2
            # do not exceed `N`
            width = min(T_est, N)
        else:
            width = np.argmin(np.abs(ratio - fast_approx_amp_ratio))
        return width

    # if complete decay, search within length's scale
    support = 2 * compute_temporal_support(p_f, **ca)
    complete_decay = bool(support != Np)
    too_short = bool(N == 2 or Np == 2)
    if too_short:
        return (1 if complete_decay else 2)
    elif not complete_decay:
        # cannot exceed by definition
        T_max = N
        # if it were less, we'd have `complete_decay`
        T_min = 2 ** math.ceil(math.log2(Np / complete_decay_factor))
    else:  # complete decay
        # if it were more, we'd have `not complete_decay`
        # the `+ 1` is to be safe in edge cases
        T_max = 2 ** math.ceil(math.log2(Np / complete_decay_factor) + 1)
        # follows from relation of `complete_decay_factor` to `width`
        # `width \propto support`, `support = complete_decay_factor * stuff`
        # (asm. full decay); so approx `support ~= complete_decay_factor * width`
        T_min = 2 ** math.floor(math.log2(support / complete_decay_factor))
    T_min = max(min(T_min, T_max // 2), 1)  # ensure max > min and T_min >= 1
    T_max = max(T_max, 2)  # ensure T_max >= 2
    T_min_orig, T_max_orig = T_min, T_max

    n_scales = math.log2(T_max) - math.log2(T_min)
    search_pts = int(n_scales * pts_per_scale)

    # search T ###############################################################
    def search_T(T_min, T_max, search_pts, log):
        Ts = (np.linspace(T_min, T_max, search_pts) if not log else
              np.logspace(np.log10(T_min), np.log10(T_max), search_pts))
        Ts = np.unique(np.round(Ts).astype(int))

        Ts_done = []
        corrs = []
        for T_test in Ts:
            N_phi = max(int(T_test * complete_decay_factor), Np)
            phi_f = gauss_1d(N_phi, sigma=sigma0 / T_test)
            phi_t = ifft(phi_f).real

            trim = min(min(len(p_t), len(phi_t))//2, N)
            p0, p1 = p_t[:trim], phi_t[:trim]
            p0 /= np.linalg.norm(p0)  # /= sqrt(sum(x**2))
            p1 /= np.linalg.norm(p1)
            corrs.append((p0 * p1).sum())
            Ts_done.append(T_test)

        T_est = int(round(Ts_done[np.argmax(corrs)]))
        T_stride = int(Ts[1] - Ts[0])
        return T_est, T_stride

    # first search in log space
    T_est, _ = search_T(T_min, T_max, search_pts, log=True)
    # refine search, now in linear space
    T_min = max(2**math.floor(math.log2(max(T_est - 1, 1))), T_min_orig)
    # +1 to ensure T_min != T_max
    T_max = min(2**math.ceil(math.log2(T_est + 1)), T_max_orig)
    # only one scale now
    search_pts = pts_per_scale
    T_est, T_stride = search_T(T_min, T_max, search_pts, log=False)
    # only within one zoom
    diff = pts_per_scale // 2
    T_min, T_max = max(T_est - diff, 1), max(T_est + diff - 1, 3)
    T_est, _ = search_T(T_min, T_max, search_pts, log=False)

    width = T_est
    return width
