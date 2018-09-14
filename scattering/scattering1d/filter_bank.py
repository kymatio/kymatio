import numpy as np
import torch
import math


def adaptative_choice_P(sigma, eps=1e-7):
    """
    Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the FFT of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of
    * a Gabor term hat psi(omega) = hat g_{sigma}(omega - xi)
    where 0 < xi < 1 is some frequency and g_{sigma} is
    the Gaussian window defined in Fourier by
    hat g_{sigma}(omega) = e^{-omega^2/(2 sigma^2)}
    * a low pass term \hat \phi which is proportional to \hat g_{\sigma}.

    If \sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the FFTs of both filters on the interval [1-P, P], the magnitude of
    the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

    |\hat \psi(P)| <= eps and |\hat \phi(1-P)| <= eps

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\sigma} in Fourier.

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


def periodize_filter_fft(h_fft, nperiods=1):
    """
    Computes a periodization of a filter provided in the FFT domain.

    Parameters
    ----------
    h_fft : array_like
        complex numpy array of shape (N*n_periods,)
    n_periods: int, optional
        Number of periods which should be used to periodize

    Returns
    -------
    v_fft : array_like
        complex numpy array of size (N,), which is a periodization of
        h_fft as described in the formula:
        v_fft[k] = sum_{i=0}^{n_periods - 1} h_fft[i * N + k]
    """
    N = h_fft.shape[0] // nperiods
    v_fft = h_fft.reshape(nperiods, N).mean(axis=0)
    return v_fft


def morlet1D(N, xi, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the FFT of a Morlet filter.

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
        the periodicity of the FFT. (At most 2*P_max - 1 periods are used,
        to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)

    Returns
    -------
    morlet_fft : array_like
        numpy array of size (N,) containing the FFT of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    # Find the adequate value of P
    P = min(adaptative_choice_P(sigma, eps=eps), P_max)
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    else:
        raise ValueError('P should be > 0, got ', P)
    # define the gabor at freq xi and the low-pass, both of width sigma
    gabor_fft = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_fft = np.exp(-(freqs_low**2) / (2 * sigma**2))
    # discretize in signal <=> periodize in Fourier
    gabor_fft = periodize_filter_fft(gabor_fft, nperiods=2 * P - 1)
    low_pass_fft = periodize_filter_fft(low_pass_fft, nperiods=2 * P - 1)
    # find the summation factor to ensure that morlet_fft[0] = 0.
    kappa = gabor_fft[0] / low_pass_fft[0]
    morlet_fft = gabor_fft - kappa * low_pass_fft
    # normalize the Morlet if necessary
    morlet_fft *= get_normalizing_factor(morlet_fft, normalize=normalize)
    return morlet_fft


def get_normalizing_factor(h_fft, normalize='l1'):
    """
    Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_fft : array_like
        numpy vector containing the FFT of a filter
    normalized : string, optional
        desired normalization type, either 'l1' or 'l2'. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that h_fft * norm_factor is the adequately normalized vector.
    """
    h_real = np.fft.ifft(h_fft)
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


def gauss1D(N, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the FFT of a low pass gaussian window.

    \hat g_{\sigma}(\omega) = e^{-\omega^2 / 2 \sigma^2}

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
        the periodicity of the FFT. (At most 2*P_max - 1 periods are used,
        to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float, optional
        required machine precision (to choose the adequate P)

    Returns
    -------
    g_fft : array_like
        numpy array of size (N,) containing the FFT of the filter (with the
        frequencies in the np.fft.fftfreq convention).
    """
    # Find the adequate value of P
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptative_choice_P(sigma, eps=eps), P_max)
    # switch cases
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    else:
        raise ValueError('P should be an integer > 0, got {}'.format(P))
    # define the low pass
    g_fft = np.exp(-freqs_low**2 / (2 * sigma**2))
    # periodize it
    g_fft = periodize_filter_fft(g_fft, nperiods=2 * P - 1)
    # normalize the signal
    g_fft *= get_normalizing_factor(g_fft, normalize=normalize)
    # return the FFT
    return g_fft


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


def compute_temporal_support(h_fft, criterion_amplitude=1e-3):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain

    This function computes the support T which is the smallest integer
    such that for all signals x and all filters h,

    \| x \conv h - x \conv h_{[-T, T]} \|_{\infty} \leq \epsilon
        \| x \|_{\infty}  (1)

    where 0<\epsilon<1 is an acceptable error, and h_{[-T, T]} denotes the
    filter h whose support is restricted in the interval [-T, T]

    The resulting value T used to pad the signals to avoid boundary effects
    and numerical errors.

    Parameters
    ----------
    h_fft : array_like
        a numpy array of size batch x time, where each row contains the
        FFT of a filter which is centered and whose absolute value is
        symmetric
    criterion_amplitude : float, optional
        value \epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_fft

    """
    h = np.fft.ifft(h_fft, axis=1)
    half_support = h.shape[1] // 2
    # compute ||h - h_[-T, T]||_1
    l1_residual = np.fliplr(
        np.cumsum(np.fliplr(np.abs(h)[:, :half_support]), axis=1))
    # find the first point above criterion_amplitude
    T = np.min(
        np.where(np.max(l1_residual, axis=0) <= criterion_amplitude)[0]) + 1
    return T


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
    n = cv['key'][1]
    new_cv = {'xi': cv['xi'] * factor, 'sigma': cv['sigma'] * factor}
    # compute the new j
    j = get_max_dyadic_subsampling(new_cv['xi'], new_cv['sigma'], alpha=alpha)
    new_cv['key'] = (j, n + 1)
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

    xi = {}
    sigma = {}

    if sigma_max <= sigma_low:
        # in this exceptional case, we will not go through the loop, so
        # we directly assign
        last_xi = sigma_max
        n = 0
    else:
        # fill all the dyadic wavelets as long as possible
        current = {'key': (0, 0), 'xi': xi_max, 'sigma': sigma_max}
        while current['sigma'] > sigma_low:  # while we can attribute something
            xi[current['key']] = current['xi']
            sigma[current['key']] = current['sigma']
            current = move_one_dyadic_step(current, Q, alpha=alpha)
        # get the last key
        last_key = max(xi.keys())
        n = last_key[1] + 1
        last_xi = xi[last_key]
    # fill num_interm wavelets between last_xi and 0, both excluded
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_low
        j = get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha)
        key = (j, n)
        xi[key] = new_xi
        sigma[key] = new_sigma
        n += 1
    # return results
    return xi, sigma


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
    xi1, sigma1 = compute_params_filterbank(sigma_low, Q, r_psi=r_psi,
                                            alpha=alpha)
    xi2, sigma2 = compute_params_filterbank(sigma_low, 1, r_psi=r_psi,
                                            alpha=alpha)
    return sigma_low, xi1, sigma1, xi2, sigma2


def scattering_filter_factory(J_support, J_scattering, Q, r_psi=math.sqrt(0.5),
                              criterion_amplitude=1e-3, normalize='l1',
                              to_torch=False, max_subsampling=None,
                              sigma0=0.1, alpha=5., P_max=5, eps=1e-7,
                              **kwargs):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * k where k is an integer bounded below by 0. The maximal value for k
        depends on the type of filter, it is dynamically chosen depending
        on max_subsampling and the characteristics of the filters.
        Each value for k is an array (or tensor) of size 2**(J_support - k)
        containing the FFT of the filter after subsampling by 2**k

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
    to_torch: boolean, optional
        whether the filters should be provided as torch
        tensors (true) or numpy arrays (false). Defaults to False.
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
        maximal number of periods to use to make sure that the
        FFT of the filters is periodic. P_max = 5 is more than enough for
        double precision. Defaults to 5. Should be >= 1
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to 1e-7

    Returns
    -------
    phi_fft : dictionary
        a dictionary containing the low-pass filter at all possible
        subsamplings. See above for a description of the dictionary structure.
        The possible subsamplings are controlled by the inputs they can
        receive, which correspond to the subsamplings performed on top of the
        1st and 2nd order transforms.
    psi1_fft : dictionary
        a dictionary containing the band-pass filters of the 1st order,
        only for the base resolution as no subsampling is used in the
        scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of the type (j, n) where n is an
        integer counting the filters and j the maximal dyadic subsampling
        which can be performed on top of the filter without aliasing.
    psi2_fft : dictionary
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
    sigma_low, xi1, sigma1, xi2, sigma2 = calibrate_scattering_filters(
        J_scattering, Q, r_psi=r_psi, sigma0=sigma0, alpha=alpha)

    # instantiate the dictionaries which will contain the filters
    phi_fft = {}
    psi1_fft = {k: {} for k in xi1.keys()}
    psi2_fft = {k: {} for k in xi2.keys()}

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    max_subsampling_after_psi2 = 0
    for key in xi2.keys():
        j2 = key[0]
        # compute the current value for the max_subsampling,
        # which depends on the input it can accept.
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [
                j1 for (j1, n1) in xi1.keys() if j2 > j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling
        # save it for later use
        max_subsampling_after_psi2 = max(max_subsampling_after_psi2,
                                         max_sub_psi2 + j2)
        # compute the filter after subsampling at all subsamplings
        # which might be received by the network
        for subsampling in range(0, max_sub_psi2 + 1):
            T = 2**(J_support - subsampling)
            for key in xi2.keys():
                psi2_fft[key][subsampling] = morlet1D(
                    T, xi2[key], sigma2[key], normalize=normalize,
                    P_max=P_max, eps=eps)

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with T=2**J_support
    for key in xi1.keys():
        T = 2**J_support
        psi1_fft[key][0] = morlet1D(
            T, xi1[key], sigma1[key], normalize=normalize,
            P_max=P_max, eps=eps)

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    if max_subsampling is None:
        max_subsampling_after_psi1 = max([key[0] for key in psi1_fft.keys()])
        max_sub_phi = max(max_subsampling_after_psi1,
                          max_subsampling_after_psi2)
    else:
        max_sub_phi = max_subsampling
    # compute the filters at all possible subsamplings
    for subsampling in range(0, max_sub_phi + 1):
        T = 2**(J_support - subsampling)
        # compute the low_pass filter
        phi_fft[subsampling] = gauss1D(T, sigma_low, P_max=P_max, eps=eps)

    # Embed the meta information within the filters
    for k in xi1.keys():
        psi1_fft[k]['xi'] = xi1[k]
        psi1_fft[k]['sigma'] = sigma1[k]
    for k in xi2.keys():
        psi2_fft[k]['xi'] = xi2[k]
        psi2_fft[k]['sigma'] = sigma2[k]
    phi_fft['xi'] = 0.
    phi_fft['sigma'] = sigma_low

    # compute the support size allowing to pad without boundary errors
    # at the finest resolution
    t_max_phi = compute_temporal_support(
        phi_fft[0].reshape(1, -1), criterion_amplitude=criterion_amplitude)

    # prepare for pytorch if necessary
    if to_torch:
        for k in phi_fft.keys():
            if type(k) != str:
                # view(-1, 1) because real numbers!
                phi_fft[k] = torch.from_numpy(phi_fft[k]).view(-1, 1)
        for k in psi1_fft.keys():
            for sub_k in psi1_fft[k].keys():
                if type(sub_k) != str:
                    # view(-1, 1) because real numbers!
                    psi1_fft[k][sub_k] = torch.from_numpy(
                        psi1_fft[k][sub_k]).view(-1, 1)
        for k in psi2_fft.keys():
            for sub_k in psi2_fft[k].keys():
                if type(sub_k) != str:
                    # view(-1, 1) because real numbers!
                    psi2_fft[k][sub_k] = torch.from_numpy(
                        psi2_fft[k][sub_k]).view(-1, 1)

    # return results
    return phi_fft, psi1_fft, psi2_fft, t_max_phi
