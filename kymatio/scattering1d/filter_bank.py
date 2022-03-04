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


def periodize_filter_fourier(h_f, nperiods=1, aggregation='sum'):
    """
    Computes a periodization of a filter provided in the Fourier domain.
    Parameters
    ----------
    h_f : array_like
        complex numpy array of shape (N*n_periods,)
    n_periods: int, optional
        Number of periods which should be used to periodize
    aggregation: str['sum', 'mean'], optional
        'sum' will multiply subsampled time-domain signal by subsampling
        factor to conserve energy during scattering (rather not double-account
        for it since we already subsample after convolving).
        'mean' will only subsample the input.

    Returns
    -------
    v_f : array_like
        complex numpy array of size (N,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * N + k]
    """
    N = h_f.shape[0] // nperiods
    h_f_re = h_f.reshape(nperiods, N)
    v_f = (h_f_re.sum(axis=0) if aggregation == 'sum' else
           h_f_re.mean(axis=0))
    v_f = v_f if h_f.ndim == 1 else v_f[:, None]  # preserve dim
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
    normalize = normalize.split('-')[0]  # in case of `-energy`
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


def compute_temporal_support(h_f, criterion_amplitude=1e-3, warn=False):
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
    warn: bool (default False)
        Whether to raise a warning upon `h_f` leading to boundary effects.

    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_f

    """
    if h_f.shape[-1] == 1:
        return 1
    elif h_f.ndim == 1:
        h_f = h_f[None]

    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    # check if any value in half of worst case of abs(h) is below criterion
    hhalf = np.max(np.abs(h[:, :half_support]), axis=0)
    max_amplitude = hhalf.max()
    meets_criterion_idxs = np.where(hhalf <= criterion_amplitude * max_amplitude
                                    )[0]
    if len(meets_criterion_idxs) != 0:
        # if it is possible
        N = meets_criterion_idxs.min() + 1
        # in this case pretend it's 1 less so external computations don't
        # have to double support since this is close enough
        if N == half_support:
            N -= 1
    else:
        # if there are none
        N = half_support
        if warn:
            # Raise a warning to say that there will be border effects
            warnings.warn('Signal support is too small to avoid border effects')
    return N


def compute_minimum_required_length(fn, N_init, max_N=None,
                                    criterion_amplitude=1e-3):
    """Computes minimum required number of samples for `fn(N)` to have temporal
    support less than `N`, as determined by `compute_temporal_support`.

    Parameters
    ----------
    fn: FunctionType
        Function / lambda taking `N` as input and returning a filter in
        frequency domain.
    N_init: int
        Initial input to `fn`, will keep doubling until `N == max_N` or
        temporal support of `fn` is `< N`.
    max_N: int / None
        See `N_init`; if None, will raise `N` indefinitely.
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    N: int
        Minimum required number of samples for `fn(N)` to have temporal
        support less than `N`.
    """
    N = 2**math.ceil(math.log2(N_init))  # ensure pow 2
    while True:
        try:
            p_fr = fn(N)
        except ValueError as e:  # get_normalizing_factor()
            if "division" not in str(e):
                raise e
            N *= 2
            continue

        p_halfwidth = compute_temporal_support(
            p_fr, criterion_amplitude=criterion_amplitude, warn=False)

        if N > 1e9:  # avoid crash
            raise Exception("couldn't satisfy stop criterion before `N > 1e9`; "
                            "check `fn`")
        if 2 * p_halfwidth < N or (max_N is not None and N > max_N):
            break
        N *= 2
    return N


def get_max_dyadic_subsampling(xi, sigma, alpha=4.):
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
        The larger alpha, the smaller the error. Defaults to 4.

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


def move_one_dyadic_step(cv, Q, alpha=4.):
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
        the more conservative the algorithm is. Defaults to 4.

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
    xi_max = max(1. / (1. + math.pow(2., 1. / Q)), 0.4)
    return xi_max


def compute_params_filterbank(sigma_min, Q, r_psi=math.sqrt(0.5), alpha=4.,
                              xi_min=None):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    width of adjacent filters, up to a minimum frequency where the frequencies
    are translated. sigma_min specifies the smallest frequential width
    among all filters, while preserving the coverage of the whole frequency
    axis.

    The keys of the dictionaries are tuples of integers (j, n) where n is a
    counter (starting at 0 for the highest frequency filter) and j is the
    maximal dyadic subsampling accepted by this filter.

    Parameters
    ----------
    sigma_min : float
        This acts as a lower-bound on the frequential widths of the band-pass
        filters. The low-pass filter may be wider (if T < 2**J_scattering), making
        invariants over shorter time scales than longest band-pass filter.
    Q : int
        number of wavelets per octave.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger r_psi improves it).
        Defaults to sqrt(0.5).
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    xi_min : float, optional
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

    Returns
    -------
    xi : list[float]
        list containing the central frequencies of the wavelets.
    sigma : list[float]
        list containing the frequential widths of the wavelets.
    j : list[int]
        list containing the subsampling factors of the wavelets (closely
        related to their dyadic scales)
    is_cqt : list[bool]
        list containing True if a wavelet was built per Constant Q Transform
        (fixed `xi / sigma`), else False for the STFT portion

    Refs
    ----
    Convolutional operators in the time-frequency domain, 2.1.3, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    xi_min = xi_min if xi_min is not None else -1
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    xi = []
    sigma = []
    j = []
    is_cqt = []

    if sigma_max <= sigma_min or xi_max <= xi_min:
        # in this exceptional case, we will not go through the loop, so
        # we directly assign
        last_xi = sigma_max
    else:
        # fill all the dyadic wavelets as long as possible
        current = {'key': 0, 'j': 0, 'xi': xi_max, 'sigma': sigma_max}
        # while we can attribute something
        while current['sigma'] > sigma_min and current['xi'] > xi_min:
            xi.append(current['xi'])
            sigma.append(current['sigma'])
            j.append(current['j'])
            is_cqt.append(True)
            current = move_one_dyadic_step(current, Q, alpha=alpha)
        # get the last key
        last_xi = xi[-1]
    # fill num_interm wavelets between last_xi and 0, both excluded
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_min
        if new_xi < xi_min:
            break
        xi.append(new_xi)
        sigma.append(new_sigma)
        j.append(get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha))
        is_cqt.append(False)
    # return results
    return xi, sigma, j, is_cqt


def calibrate_scattering_filters(J, Q, T, r_psi=math.sqrt(0.5), sigma0=0.1,
                                 alpha=4., xi_min=None):
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
    Q : int / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `2` or `1` for most applications.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float / tuple[float], optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger r_psi improves it).
        Defaults to sqrt(0.5).
        Tuple sets separately for first- and second-order filters.
    sigma0 : float, optional
        frequential width of the low-pass filter at scale J=0
        (the subsequent widths are defined by sigma_J = sigma0 / 2^J).
        Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    xi_min : float, optional
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

    Returns
    -------
    sigma_low : float
        frequential width of the low-pass filter
    xi1 : list[float]
        Center frequencies of the first order filters.
    sigma1 : list[float]
        Frequential widths of the first order filters.
    j1 : list[int]
        Subsampling factors of the first order filters.
    is_cqt1 : list[bool]
        Constant Q Transform construction flag of the first order filters.
    xi2, sigma2, j2, is_cqt2 :
        `xi1, sigma1, j1, is_cqt1` for second order filters.
    """
    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    r_psi1, r_psi2 = r_psi if isinstance(r_psi, tuple) else (r_psi, r_psi)
    if Q1 < 1 or Q2 < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))

    # lower bound of band-pass filter frequential widths:
    # for default T = 2**(J), this coincides with sigma_low
    sigma_min = sigma0 / math.pow(2, J)

    xi1s, sigma1s, j1s, is_cqt1s = compute_params_filterbank(
        sigma_min, Q1, r_psi=r_psi1, alpha=alpha, xi_min=xi_min)
    xi2s, sigma2s, j2s, is_cqt2s = compute_params_filterbank(
        sigma_min, Q2, r_psi=r_psi2, alpha=alpha, xi_min=xi_min)

    # width of the low-pass filter
    sigma_low = sigma0 / T
    return sigma_low, xi1s, sigma1s, j1s, is_cqt1s, xi2s, sigma2s, j2s, is_cqt2s


def scattering_filter_factory(J_support, J_scattering, Q, T,
                              r_psi=math.sqrt(0.5),
                              criterion_amplitude=1e-3, normalize='l1',
                              max_subsampling=None, sigma0=0.1, alpha=4.,
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
    * 'support': temporal support

    Parameters
    ----------
    J_support : int
        2**J_support is the desired support size of the filters
    J_scattering : int
        parameter for the scattering transform (2**J_scattering
        corresponds to maximal temporal support of any filter)
    Q : int >= 1 / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `1` for most (`Scattering1D)` applications.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float / tuple[float], optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger r_psi improves it).
        Defaults to sqrt(0.5).
        Tuple sets separately for first- and second-order filters.
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
        subsampling is. Defaults to 4.
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

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    N = 2**J_support
    xi_min = 2 / N  # minimal peak at bin 2
    # compute the spectral parameters of the filters
    (sigma_low, xi1s, sigma1s, j1s, is_cqt1s, xi2s, sigma2s, j2s, is_cqt2s
     ) = calibrate_scattering_filters(J_scattering, Q, T, r_psi=r_psi,
                                      sigma0=sigma0, alpha=alpha, xi_min=xi_min)

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
        psi_f = {}
        psi_f[0] = morlet_1d(
            N, xi2s[n2], sigma2s[n2], normalize=normalize, P_max=P_max, eps=eps)
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
        psi1_f.append({0: morlet_1d(
            N, xi1s[n1], sigma1s[n1], normalize=normalize, P_max=P_max, eps=eps)})

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
    phi_f[0] = gauss_1d(N, sigma_low, P_max=P_max, eps=eps)
    for subsampling in range(1, max_sub_phi + 1):
        factor_subsampling = 2**subsampling
        # compute the low_pass filter
        phi_f[subsampling] = periodize_filter_fourier(
            phi_f[0], nperiods=factor_subsampling)

    # Embed the meta information within the filters
    ca = dict(criterion_amplitude=criterion_amplitude)
    s0ca = dict(N=N, sigma0=sigma0, criterion_amplitude=criterion_amplitude)
    for (n1, j1) in enumerate(j1s):
        psi1_f[n1]['xi'] = xi1s[n1]
        psi1_f[n1]['sigma'] = sigma1s[n1]
        psi1_f[n1]['j'] = j1
        psi1_f[n1]['is_cqt'] = is_cqt1s[n1]
        psi1_f[n1]['width'] = {0: 2*compute_temporal_width(
            psi1_f[n1][0], **s0ca)}
        psi1_f[n1]['support'] = {0: 2*compute_temporal_support(
            psi1_f[n1][0], **ca)}

    for (n2, j2) in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2s[n2]
        psi2_f[n2]['sigma'] = sigma2s[n2]
        psi2_f[n2]['j'] = j2
        psi2_f[n2]['is_cqt'] = is_cqt2s[n2]
        psi2_f[n2]['width'] = {}
        psi2_f[n2]['support'] = {}
        for k in psi2_f[n2]:
            if isinstance(k, int):
                psi2_f[n2]['width'][k] = 2*compute_temporal_width(
                    psi2_f[n2][k], **s0ca)
                psi2_f[n2]['support'][k] = 2*compute_temporal_support(
                    psi2_f[n2][k], **ca)

    phi_f['xi'] = 0.
    phi_f['sigma'] = sigma_low
    phi_f['j'] = log2_T
    phi_f['width'] = 2*compute_temporal_width(phi_f[0], **s0ca)
    phi_f['support'] = 2*compute_temporal_support(phi_f[0], **ca)

    # return results
    return phi_f, psi1_f, psi2_f


def psi_fr_factory(J_pad_frs_max_init, J_fr, Q_fr, N_frs, N_fr_scales_max,
                   N_fr_scales_min, max_pad_factor_fr, unrestricted_pad_fr,
                   max_subsample_equiv_before_phi_fr,
                   subsample_equiv_relative_to_max_pad_init,
                   average_fr_global_phi,
                   sampling_psi_fr='resample', sampling_phi_fr='resample',
                   pad_mode_fr='conj-reflect-zero',
                   sigma_max_to_min_max_ratio=1.2, r_psi_fr=math.sqrt(0.5),
                   normalize_fr='l1', criterion_amplitude=1e-3, sigma0=0.1,
                   alpha=4., P_max=5, eps=1e-7):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency
    * 'sigma': frequential width
    * 'j': subsampling factor from 0 to `J_fr` (or potentially less if
      `sampling_psi_fr != 'resample'`).
    * 'width': temporal width (scale; interval of imposed invariance)
    * 'support': temporal support (duration of decay)

    Parameters
    ----------
    J_pad_frs_max_init : int
        `2**J_pad_frs_max_init` is the largest length of all filters.

    J_fr : int
        The maximum log-scale of frequential scattering in joint scattering
        transform, and number of octaves of frequential filters. That is,
        the maximum (bandpass) scale is given by :math:`2^J_fr`.

    Q_fr : int
        Number of wavelets per octave for frequential scattering.

    N_frs, N_fr_scales_max, N_fr_scales_min : list[int], int, int
        See `help(TimeFrequencyScattering1D)`. Used for filter computation at
        various lengths/scales depending on `sampling_psi_fr`.

    max_pad_factor_fr : list[int] / None
        See `help(TimeFrequencyScattering1D)`. Restricts filter length, but
        may be overridden.

    unrestricted_pad_fr : bool
        `== max_pad_factor is None`. `True` adds a few checks to ensure filters
        are constructed fully correctly and devoid of boundary effects.

    max_subsample_equiv_before_phi_fr : int
        See `help(TimeFrequencyScattering1D)`. Used for waiving certain criterion
        checks if `N_fr_scales` is too small.

    subsample_equiv_relative_to_max_pad_init : int
        See `help(TimeFrequencyScattering1D)`. Controls filter lengths.

    average_fr_global_phi : bool
        See `help(TimeFrequencyScattering1D)`. Used for waiving certain criterion
        checks.

    sampling_psi_fr : str['resample', 'recalibrate', 'exclude']
        See `help(TimeFrequencyScattering1D)`.
        In terms of effect on maximum `j` per `n1_fr`:

            - 'resample': no variation (by design, all temporal properties are
              preserved, including subsampling factor).
            - 'recalibrate': `j1_fr_max` is (likely) lesser with greater
              `subsample_equiv_due_to_pad` (by design, temporal width is halved
              for shorter `N_frs`). The limit, however, is set by
              `sigma_max_to_min_max_ratio` (see its docs).
            - 'exclude': approximately same as 'recalibrate'. By design, excludes
              temporal widths above `min_width * 2**downsampling_factor`, which
              is likely to reduce `j1_fr_max` with greater
              `subsample_equiv_due_to_pad`.
                - It's "approximately" same because center frequencies and
                  widths are different; depending how loose our alias tolerance
                  (`alpha`), they're exactly the same.

    sampling_phi_fr : str['resample', 'recalibrate']
        See `help(TimeFrequencyScattering1D)`. Used for a sanity check.

    pad_mode_fr : str
        See `help(TimeFrequencyScattering1D)`. Used for a sanity check.

    sigma_max_to_min_max_ratio : float
        Largest permitted `max(sigma) / min(sigma)`.
        See `help(TimeFrequencyScattering1D)`.

    r_psi_fr, normalize_fr, criterion_amplitude, sigma0, alpha, P_max, eps:
        See `help(kymatio.scattering1d.filter_bank.scattering_filter_factory)`.

    Returns
    -------
    psi1_f_fr_up : list[dict]
        List of dicts containing the band-pass filters of frequential scattering
        with "up" spin at all possible downsamplings. Each element corresponds to
        a dictionary for a single filter, see above for an exact description.
        The `'j'` key holds the value of maximum subsampling which can be
        performed on each filter without aliasing.

        These filters are not subsampled, as they do not receive subsampled
        inputs (but their *outputs*, i.e. convolutions, can be subsampled).
        Downsampling is always a *trimming*, not a subsampling, and amount
        is determined by filter `xi` and `sigma`, controlled by `sampling_psi_fr`.
        Downsampling is indexed by `j` (rather, `subsample_equiv_due_to_pad`).

        Example (`J_fr = 2`, `n1_fr = 8`, lists hold subsampling factors):
            - 'resample':
                0: [2, 1, 0]
                1: [2, 1, 0]
                2: [2, 1, 0]
            - 'recalibrate':
                0: [2, 1, 0]
                1: [1, 1, 0]
                2: [0, 0, 0]
            - 'exclude':
                0: [2, 1, 0]
                1: [1, 0]
                2: [0]

    psi1_f_fr_down : list[dict]
        Same as `psi1_f_fr_up` but with "down" spin (analytic, whereas "up"
        is anti-analytic wavelet).

    j0_max : int / None
        Sets `max_subsample_equiv_before_psi_fr`, see its docs in
        `help(TimeFrequencyScattering1D)`.

    Note
    ----
    For `average_fr_global==True`, largest `j0` for `psi_fr` may exceed that of
    `phi_fr`, since `max_subsample_equiv_before_phi_fr` is set to
    `J_pad_frs_max_init` rather than largest `j0` in `phi_fr`. This is for speed,
    and since `phi_fr` will never be used.
    """
    # compute the spectral parameters of the filters
    J_support = J_pad_frs_max_init  # begin with longest
    N = 2**J_support
    xi_min = 2 / N  # minimal peak at bin 2
    T = 1  # for computing `sigma_low`, unused
    (_, xi1_frs, sigma1_frs, j1_frs, is_cqt1_frs, *_
     ) = calibrate_scattering_filters(J_fr, Q_fr, T=T, r_psi=r_psi_fr,
                                      sigma0=sigma0, alpha=alpha, xi_min=xi_min)

    # instantiate the dictionaries which will contain the filters
    psi1_f_fr_up = []
    psi1_f_fr_down = []

    j0_max, scale_diff_max = None, None
    ca = dict(criterion_amplitude=criterion_amplitude)
    s0ca = dict(criterion_amplitude=criterion_amplitude, sigma0=sigma0)
    if sampling_psi_fr == 'recalibrate':
        # recalibrate filterbank to each j0
        (xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new, scale_diff_max
         ) = _recalibrate_psi_fr(xi1_frs, sigma1_frs, j1_frs, is_cqt1_frs, N,
                                 alpha, N_fr_scales_min, N_fr_scales_max,
                                 sigma_max_to_min_max_ratio)
    elif sampling_psi_fr == 'resample' and unrestricted_pad_fr:
        # in this case filter temporal behavior is preserved across all lengths
        # so we must restrict lowest length such that widest filter still decays
        j0 = 0
        while True:
            psi_widest = morlet_1d(N // 2**j0, xi1_frs[-1], sigma1_frs[-1],
                                   P_max=P_max, normalize=normalize_fr, eps=eps
                                   )[:, None]
            psi_widest_support = 2*compute_temporal_support(psi_widest.T, **ca)
            if psi_widest_support == len(psi_widest):
                j0_max = j0 - 1
                # in zero padding we cut padding in half, which distorts
                # the wavelet but negligibly relative to the scattering scale
                if j0_max < 0 and pad_mode_fr != 'zero':
                    raise Exception("got `j0_max = %s < 0`, meaning " % j0_max
                                    + "`J_pad_frs_max_init` computed incorrectly.")
                j0_max = max(j0_max, 0)
                break
            elif len(psi_widest) == N_fr_scales_min:
                # smaller pad length is impossible
                break
            j0 += 1
    elif sampling_psi_fr == 'exclude':
        # this is built precisely to enable `j0_max=None` while preserving
        # temporal behavior (so long as there's at least one filter remaining;
        # j0_max is set later)
        j0_max_exclude = {}
        pass

    def get_params(n1_fr, scale_diff):
        if sampling_psi_fr in ('resample', 'exclude'):
            return (xi1_frs[n1_fr], sigma1_frs[n1_fr], j1_frs[n1_fr],
                    is_cqt1_frs[n1_fr])
        elif sampling_psi_fr == 'recalibrate':
            return (xi1_frs_new[scale_diff][n1_fr],
                    sigma1_frs_new[scale_diff][n1_fr],
                    j1_frs_new[scale_diff][n1_fr],
                    is_cqt1_frs_new[scale_diff][n1_fr])

    # keep a mapping from `j0` to `scale_diff`
    j0_to_scale_diff = {}
    # for later
    same_pad_limit = (unrestricted_pad_fr or
                      all(p == max_pad_factor_fr[0] for p in max_pad_factor_fr))
    pad_contractive_phi = (average_fr_global_phi or
                           sampling_phi_fr == 'recalibrate')
    cleanup = False  # for later
    # sample spin down and up wavelets
    for n1_fr in range(len(j1_frs)):
        psi_down = {}
        # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
        psi_down[0] = morlet_1d(N, xi1_frs[n1_fr], sigma1_frs[n1_fr],
                                normalize=normalize_fr, P_max=P_max, eps=eps
                                )[:, None]
        psi_down['width'] = {0: 2*compute_temporal_width(
            psi_down[0], N=2**N_fr_scales_max, **s0ca)}
        psi_down['support'] = {0: 2*compute_temporal_support(psi_down[0].T, **ca)}

        # j0 is ordered greater to lower, so reverse
        j0_prev = 0
        for j0, N_fr in zip(subsample_equiv_relative_to_max_pad_init[::-1],
                            N_frs[::-1]):
            #### Validate `j0` & compute scale params ########################
            # ensure we compute at valid `j0`
            if j0 < 0:
                continue
            # `j0_max` restricts the *least* we can pad by for any `N_fr`,
            # while `max_pad_factor_fr` restricts the *most*.
            # Latter takes precedence.
            elif j0_max is not None and j0 > j0_max:
                # subsequent `j0` is only greater (if not, Exception will be
                # raised in `adjust_padding_and_filters()`)
                break

            # compute scale params
            N_fr_scales = math.ceil(math.log2(N_fr))
            scale_diff = N_fr_scales_max - N_fr_scales
            if scale_diff_max is not None and scale_diff > scale_diff_max:
                # subsequent `scale_diff` are only greater
                # This takes precedence over `max_pad_factor_fr`.
                j0_max = j0_prev
                break

            # validate `subsample_equiv_relative_to_max_pad_init` ############
            # needed for variable list `max_pad_factor_fr` ###################
            # j0 no longer strictly tied to N_fr for logics that check it
            # (that raise errors); account for this
            j0_at_limit = bool(scale_diff >= max_subsample_equiv_before_phi_fr)
            must_be_unique = bool(pad_contractive_phi and same_pad_limit and
                                  not j0_at_limit)

            if j0 not in j0_to_scale_diff:
                j0_to_scale_diff[j0] = scale_diff
            # ensure every `scale_diff` maps to one `j0`
            if list(j0_to_scale_diff.values()).count(scale_diff) > 1:
                raise Exception(("same `scale_diff` yielded multiple `J_pad_fr`"
                                 "\n{}").format(j0_to_scale_diff))
            # ensure every `j0` maps to one `scale_diff`, conditionally
            # See `help(compute_padding_fr)`.
            elif j0_to_scale_diff[j0] != scale_diff:
                err = Exception(("same `J_pad_fr` mapped to different "
                                 "`scale_diff` ({})\n{}").format(
                                     scale_diff, j0_to_scale_diff))
                if sampling_psi_fr == 'exclude':
                    if must_be_unique:
                        # should not occur^1
                        # ^1: `min_to_pad` should halve with each
                        # lesser `N_fr_scales`, guaranteeing unique padding
                        raise err
                    else:
                        # would require triple-indexing
                        # Additionally, with 'resample',
                        # `width(phi) > N_fr_scales`, can't use any shorter psi
                        # `continue` because we still need the remaining `j0`
                        j0_prev = j0
                        continue
                elif sampling_psi_fr == 'recalibrate':
                    if must_be_unique:
                        # should not occur^1 with `same_pad_limit`;
                        # not allowed either way per requiring triple-indexing
                        raise err
                    else:
                        # `continue` because can still fill the remaining `j0`
                        j0_prev = j0
                        continue

            # if checks pass, our logic is correct; now ensure no recomputation
            if j0 == j0_prev:
                continue

            #### Compute wavelet #############################################
            # fetch wavelet params, sample wavelet, compute its spatial width
            xi, sigma, *_ = get_params(n1_fr, scale_diff)
            try:
                psi = morlet_1d(N // 2**j0, xi, sigma, normalize=normalize_fr,
                                P_max=P_max, eps=eps)[:, None]
            except ValueError as e:
                if "division" not in str(e):
                    raise e
                elif sampling_psi_fr == 'resample':
                    j0_max = j0_prev
                    cleanup = True
                    break
                raise e

            psi_width = 2*compute_temporal_width(psi, N=2**N_fr_scales, **s0ca)
            if sampling_psi_fr == 'exclude':
                # if wavelet exceeds max possible width at this scale, exclude it
                if psi_width > 2**N_fr_scales:
                    # subsequent `N_fr_scales` are only lesser, and `psi_width`
                    # doesn't change (approx w/ discretization error).
                    # `j0_max_exclude` can compute from `n1_fr=0` case alone
                    j0_max_exclude[n1_fr] = j0_prev
                    break

            psi_down[j0] = psi
            psi_down['width'][j0] = psi_width
            psi_down['support'][j0] = 2*compute_temporal_support(psi, **ca)
            j0_prev = j0

        psi1_f_fr_down.append(psi_down)
        # compute spin up
        psi_up = {}
        j0s = [j0 for j0 in psi_down if isinstance(j0, int)]
        for j0 in psi_down:
            if isinstance(j0, int):
                # compute spin up by conjugating spin down in frequency domain
                psi_up[j0] = conj_fr(psi_down[j0])
        psi_up['width'] = psi_down['width'].copy()
        psi_up['support'] = psi_down['support'].copy()
        psi1_f_fr_up.append(psi_up)

    # construction terminated early; remove unused `j0`
    if cleanup:
        for psi_fs in (psi1_f_fr_up, psi1_f_fr_down):
            for psi_f in psi_fs:
                j0s = [j0 for j0 in psi_f if isinstance(j0, int)]
                for j0 in j0s:
                    if j0 > j0_max:
                        del psi_f[j0]

    # Embed meta information within the filters
    for (n1_fr, j1_fr) in enumerate(j1_frs):
        for psi_f in (psi1_f_fr_down, psi1_f_fr_up):
            # create initial meta
            meta = {'xi': xi1_frs[n1_fr], 'sigma': sigma1_frs[n1_fr], 'j': j1_fr,
                    'is_cqt': is_cqt1_frs[n1_fr]}
            for field, value in meta.items():
                psi_f[n1_fr][field] = {0: value}
            # fill for j0s
            j0s = [k for k in psi_f[n1_fr] if (isinstance(k, int) and k != 0)]
            for j0 in j0s:
                xi, sigma, j, is_cqt = get_params(n1_fr, j0_to_scale_diff[j0])
                psi_f[n1_fr]['xi'][j0] = xi
                psi_f[n1_fr]['sigma'][j0] = sigma
                psi_f[n1_fr]['j'][j0] = j
                psi_f[n1_fr]['is_cqt'][j0] = is_cqt

    # to ensure at least one wavelet for every `N_fr_scales`
    if sampling_psi_fr == 'exclude' and 0 in j0_max_exclude:
        j0_max = max(j0_max_exclude.values())
        # `n1_fr==0` should be the most trimmable (lowest time width)
        assert j0_max == j0_max_exclude[0], (j0_max, j0_max_exclude)
    # ensure non-negative
    if j0_max is not None:
        assert j0_max >= 0, j0_max

    # return results
    return psi1_f_fr_up, psi1_f_fr_down, j0_max


def phi_fr_factory(J_pad_frs_max_init, F, log2_F, N_fr_scales_min,
                   unrestricted_pad_fr, sampling_phi_fr='resample',
                   criterion_amplitude=1e-3, sigma0=0.1, P_max=5, eps=1e-7):
    """
    Builds in Fourier the lowpass Gabor filters used for JTFS.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * 'j': subsampling factor from 0 to `log2_F` (or potentially less if
      `sampling_phi_fr = 'recalibrate'`).
    * 'width': temporal width (scale; interval of imposed invariance)
    * 'support': temporal support (duration of decay)

    Parameters
    ----------
    J_pad_frs_max_init : int
        `2**J_pad_frs_max_init` is the largest length of the filters.

    F : int
        temporal width of frequential low-pass filter, controlling amount of
        imposed frequency transposition invariance and maximum frequential
        subsampling.

    log2_F : int
        Equal to `log2(prevpow2(F))`, sets maximum subsampling factor.
        If `sampling_phi_fr=True`, this factor may not be reached *by the filter*,
        as temporal width is preserved upon resampling rather than halved as
        with subsampling. Subsampling by `log2_F` *after* convolving with
        `phi_f_fr` is fine, thus the restriction is to not subsample by more than
        the most subsampled `phi_f_fr` *before* convolving with it - set by
        `max_subsample_before_phi_fr`.

    N_fr_scales_min : int
        Used to determine the shortest filter.

    unrestricted_pad_fr : bool
        `== max_pad_factor is None`. If True, will terminate filter construction
        if lowpass decays insufficiently. Thus `max_pad_factor_fr` (not None)
        overrides boundary effect / filter distortion considerations.

    sampling_phi_fr : str['resample', 'recalibrate']
        See `help(TimeFrequencyScattering1D)`.

    criterion_amplitude, sigma, P_max, eps:
        See `help(kymatio.scattering1d.filter_bank.scattering_filter_factory)`.

    Returns
    -------
    phi_f_fr : dict[list]
        A dictionary containing the low-pass filter at all possible lengths.
        A distinction is made between input length difference due to trimming
        (or padding less) and subsampling (in frequential scattering with `psi`):
            `phi = phi_f_fr[subsample_equiv_due_to_pad][n1_fr_subsample]`
        so lists hold subsamplings of each trimming.

        Example (`log2_F = 2`, lists hold subsampling factors):
            - 'resample':
                0: [2, 1, 0]
                1: [2, 1, 0]
                2: [2, 1, 0]
                3: [2, 1, 0]
            - 'recalibrate':
                0: [2, 1, 0]
                1: [1, 0]
                2: [0]
                3: not allowed
        ('recalibrate' looks like 'exclude' for psi because it is one and the same
         filter, so it makes no sense to have e.g. `[0, 0]` (they're identical)).
    """
    # compute the spectral parameters of the filters
    sigma_low = sigma0 / F
    J_support = J_pad_frs_max_init
    N = 2**J_support

    # initial lowpass
    phi_f_fr = {}
    # expand dim to multiply along freq like (2, 32, 4) * (32, 1)
    phi_f_fr[0] = [gauss_1d(N, sigma_low, P_max=P_max, eps=eps)[:, None]]

    def compute_all_subsamplings(phi_f_fr, j0):
        for j0_sub in range(1, 1 + log2_F):
            phi_f_fr[j0].append(periodize_filter_fourier(
                phi_f_fr[j0][0], nperiods=2**j0_sub))

    compute_all_subsamplings(phi_f_fr, j0=0)

    # lowpass filters at all possible input lengths
    min_possible_pad_fr = N_fr_scales_min
    max_possible_j0 = J_pad_frs_max_init - min_possible_pad_fr
    for j0 in range(1, 1 + max_possible_j0):
        factor = 2**j0
        J_pad_fr = J_pad_frs_max_init - j0  # == N // factor
        if sampling_phi_fr == 'resample' and J_pad_fr < log2_F:
            # length is below target scale
            break
        elif sampling_phi_fr == 'recalibrate' and j0 > log2_F:
            # subsampling by more than log2_F
            break
        # ^ override `max_pad_factor_fr`

        if sampling_phi_fr == 'resample':
            prev_phi = phi_f_fr[j0 - 1][0]
            prev_phi_halfwidth = compute_temporal_support(
                prev_phi, criterion_amplitude=criterion_amplitude)

            if (prev_phi_halfwidth == prev_phi.size // 2 and
                    unrestricted_pad_fr):
                # This means width is already too great for own length,
                # so lesser length will distort lowpass.
                # Frontend will adjust "all possible input lengths" accordingly.
                # `max_pad_factor_fr` takes precedence, but adjustments will
                # still be made to exclude unnecessarily short phi.
                break

            phi_f_fr[j0] = [gauss_1d(N // factor, sigma_low, P_max=P_max,
                                     eps=eps)[:, None]]
            # dedicate separate filters for *subsampled* as opposed to *trimmed*
            # inputs (i.e. `n1_fr_subsample` vs `J_pad_frs_max_init - J_pad_fr`)
            # note this increases maximum subsampling of phi_fr relative to
            # J_pad_frs_max_init
            compute_all_subsamplings(phi_f_fr, j0=j0)
        else:
            # These won't differ from plain subsampling but we still index
            # via `subsample_equiv_relative_to_max_pad_init` and
            # `n1_fr_subsample` so just copy pointers.
            # `phi[::factor] == gauss_1d(N // factor, sigma_low * factor)`
            # when not aliased
            phi_f_fr[j0] = [phi_f_fr[0][j0_sub]
                            for j0_sub in range(j0, 1 + log2_F)]

    # embed meta info in filters
    phi_f_fr.update({field: {} for field in ('xi', 'sigma', 'j', 'width',
                                             'support')})
    j0s = [j for j in phi_f_fr if isinstance(j, int)]
    for j0 in j0s:
        xi_fr_0 = 0.
        sigma_fr_0 = (sigma_low if sampling_phi_fr == 'resample' else
                      sigma_low * 2**j0)
        j0_0 = (log2_F if sampling_phi_fr == 'resample' else
                log2_F - j0)
        for field in ('xi', 'sigma', 'j', 'support', 'width'):
            phi_f_fr[field][j0] = []
        phi_f_fr['xi'][j0] = xi_fr_0
        phi_f_fr['sigma'][j0] = sigma_fr_0
        phi_f_fr['j'][j0] = j0_0
        phi_f_fr['width'][j0] = []
        phi_f_fr['support'][j0] = []
        for j0_sub in range(len(phi_f_fr[j0])):
            # should halve with subsequent j0_sub, but compute exactly.
            # `j0`-to-`N_fr` uniqueness asserted in `psi_fr_factory`
            phi = phi_f_fr[j0][j0_sub]
            width = compute_temporal_width(
                phi, N=len(phi), sigma0=sigma0,
                criterion_amplitude=criterion_amplitude)
            support = 2*compute_temporal_support(
                phi, criterion_amplitude=criterion_amplitude)
            phi_f_fr['width'][j0].append(width)
            phi_f_fr['support'][j0].append(support)

    # return results
    return phi_f_fr


#### Energy renormalization ##################################################
def energy_norm_filterbank_tm(psi1_f, psi2_f, phi_f, J, log2_T):
    """Energy-renormalize temporal filterbank; used by `base_frontend`.
    See `help(kymatio.scattering1d.filter_bank.energy_norm_filterbank)`.
    """
    # in case of `trim_tm` for JTFS
    phi = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]
    kw = dict(phi_f=phi, J=J, log2_T=log2_T)
    psi1_f0 = [p[0] for p in psi1_f]
    psi2_f0 = [p[0] for p in psi2_f]

    energy_norm_filterbank(psi1_f0, **kw)
    scaling_factors2 = energy_norm_filterbank(psi2_f0, **kw)

    # apply unsubsampled scaling factors on subsampled
    for n2 in range(len(psi2_f)):
        for k in psi2_f[n2]:
            if isinstance(k, int) and k != 0:
                psi2_f[n2][k] *= scaling_factors2[1][n2]


def energy_norm_filterbank_fr(psi1_f_fr_up, psi1_f_fr_down, phi_f_fr,
                              J_fr, log2_F):
    """Energy-renormalize frequential filterbank; used by `base_frontend`.
    See `help(kymatio.scattering1d.filter_bank.energy_norm_filterbank)`.
    """
    # assumes same `j0` for up and down
    j0_max = max(j0 for psi_f in psi1_f_fr_up for j0 in psi_f
                 if isinstance(j0, int))

    j0_break = None
    for j0 in range(j0_max + 1):
        psi_fs_up   = [p[j0] for p in psi1_f_fr_up if j0 in p]
        psi_fs_down = [p[j0] for p in psi1_f_fr_down if j0 in p]

        if len(psi_fs_up) <= 3:  # possible with `sampling_psi_fr = 'exclude'`
            if j0 == 0:
                raise Exception("largest scale filterbank must have >=4 filters")
            j0_break = j0
            break
        phi_f = (phi_f_fr[j0][0] if j0 in phi_f_fr else
                 # `average_fr_global==True` case
                 None)
        scaling_factors = energy_norm_filterbank(
            psi_fs_up, psi_fs_down, phi_f, J_fr, log2_F)

    # reuse last's
    if j0_break is not None:
        for n1_fr in range(len(psi1_f_fr_down)):
            for j0 in range(j0_break, j0_max + 1):
                if j0 in psi1_f_fr_down[n1_fr]:
                    psi1_f_fr_up[  n1_fr][j0] *= scaling_factors[0][n1_fr]
                    psi1_f_fr_down[n1_fr][j0] *= scaling_factors[1][n1_fr]


def energy_norm_filterbank(psi_fs0, psi_fs1=None, phi_f=None, J=None, log2_T=None,
                           r_th=.3, passes=3, scaling_factors=None):
    """Rescale wavelets such that their frequency-domain energy sum
    (Littlewood-Paley sum) peaks at 2 for an analytic-only filterbank
    (e.g. time scattering for real inputs) or 1 for analytic + anti-analytic.
    This makes the filterbank energy non-expansive.

    Parameters
    ----------
    psi_fs0 : list[np.ndarray]
        Analytic filters if `psi_fs1=None`, else anti-analytic (spin up).

    psi_fs1 : list[np.ndarray] / None
        Analytic filters (spin down). If None, filterbank is treated as
        analytic-only, and LP peaks are scaled to 2 instead of 1.

    phi_f : np.ndarray / None
        Lowpass filter. If `log2_T > J`, will exclude from computation as
        it will excessively attenuate low frequency bandpasses.

    J, log2_T : int, int
        See `phi_f`. For JTFS frequential scattering these are `J_fr, log2_F`.

    r_th : float
        Redundancy threshold, determines whether "Nyquist correction" is done
        (see Algorithm below).

    passes : int
        Number of times to call this function recursively; see Algorithm.

    scaling_factors : None / dict[float]
        Used internally if `passes > 1`.

    Returns
    -------
    scaling_factors : None / dict[float]
        Used internally if `passes > 1`.

    Algorithm
    ---------
    Wavelets are scaled by maximum of *neighborhood* LP sum - precisely, LP sum
    spanning from previous to next peak location relative to wavelet of interest:
    `max(lp_sum[peak_idx[n + 1]:peak_idx[n - 1]])`. This reliably accounts for
    discretization artifacts, including the non-CQT portion.

    "Nyquist correction" is done for the highest frequency wavelet; since it
    has no "preceding" wavelet, it's its own right bound (analytic; left for
    anti-), which overestimates necessary rescaling and makes resulting LP sum
    peak above target for the *next* wavelet. This correction is done only if
    the filterbank is below a threshold redundancy (empirically determined
    `r_th=.3`), since otherwise the direct estimate is accurate.

    Multiple "passes" are done to improve overall accuracy, as not all edge
    case behavior is accounted for in one go (which is possible but complicated);
    the computational burden is minimal.
    """
    def norm_filter(psi_fs, peak_idxs, lp_sum, n, s_idx=1):
        # higher freq idx
        if n - 1 in peak_idxs:
            # midpoint
            pi0, pi1 = peak_idxs[n], peak_idxs[n - 1]
            if pi1 == pi0:
                # handle duplicate peaks
                lookback = 2
                while n - lookback in peak_idxs:
                    pi1 = peak_idxs[n - lookback]
                    if pi1 != pi0:
                        break
                    lookback += 1
            midpt = (pi0 + pi1) / 2
            a = (math.ceil(midpt) if s_idx == 1 else
                 math.floor(midpt))
        else:
            a = peak_idxs[n]

        # lower freq idx
        if n + 1 in peak_idxs:
            if n == 0 and nyquist_correction:
                b = a + 1 if s_idx == 0 else a - 1
            else:
                b = peak_idxs[n + 1]
        else:
            b = (None if s_idx == 0 else
                 1)  # exclude dc

        # peak duplicate
        if a == b:
            if s_idx == 0:
                b += 1
            else:
                b -= 1
        start, end = (a, b) if s_idx == 0 else (b, a)

        # include endpoint
        if end is not None:
            end += 1

        # if we're at endpoints, don't base estimate on single point
        if start is None:  # left endpoint
            end = max(end, 2)
        elif end is None:  # right endpoint
            start = min(start, len(lp_sum) - 1)
        elif end - start == 1:
            if start == 0:
                end += 1
            elif end == len(lp_sum) - 1:
                start -= 1

        lp_max = lp_sum[start:end].max()
        factor = np.sqrt(peak_target / lp_max)
        psi_fs[n] *= factor
        if n not in scaling_factors[s_idx]:
            scaling_factors[s_idx][n] = 1
        scaling_factors[s_idx][n] *= factor

    def correct_nyquist(psi_fs_all, peak_idxs, lp_sum):
        def _do_correction(start, end, s_idx=1):
            lp_max = lp_sum[start:end].max()
            factor = np.sqrt(peak_target / lp_max)
            for n in (0, 1):
                psi_fs[n] *= factor
                scaling_factors[s_idx][n] *= factor

        # first (Nyquist-nearest) psi rescaling may drive LP sum above bound
        # for second psi, since peak was taken only at itself
        if analytic_only:
            psi_fs = psi_fs_all
            # include endpoint
            start, end = peak_idxs[2], peak_idxs[0] + 1
            _do_correction(start, end)
        else:
            for s_idx, psi_fs in enumerate(psi_fs_all):
                a = peak_idxs[s_idx][0]
                b = peak_idxs[s_idx][2]
                start, end = (a, b) if s_idx == 0 else (b, a)
                # include endpoint
                end += 1
                _do_correction(start, end, s_idx)

    # run input checks #######################################################
    assert len(psi_fs0) >= 3, (
        "must have at least 3 filters in filterbank (got %s)" % len(psi_fs0))
    if psi_fs1 is not None:
        assert len(psi_fs0) == len(psi_fs1), (
            "analytic & anti-analytic filterbanks "
            "must have same number of filters")
    # assume same overlap for analytic and anti-analytic
    r = compute_filter_redundancy(psi_fs0[0], psi_fs0[1])
    nyquist_correction = bool(r < r_th)

    # as opposed to `analytic_and_anti_analytic`
    analytic_only = bool(psi_fs1 is None)
    peak_target = 2 if analytic_only else 1

    # execute ################################################################
    # store rescaling factors
    if scaling_factors is None:  # else means passes>1
        scaling_factors = {0: {}, 1: {}}

    # compute peak indices
    peak_idxs = {}
    if analytic_only:
        psi_fs_all = psi_fs0
        for n, psi_f in enumerate(psi_fs0):
            peak_idxs[n] = np.argmax(psi_f)
    else:
        psi_fs_all = (psi_fs0, psi_fs1)
        for s_idx, psi_fs in enumerate(psi_fs_all):
            peak_idxs[s_idx] = {}
            for n, psi_f in enumerate(psi_fs):
                peak_idxs[s_idx][n] = np.argmax(psi_f)

    # ensure LP sum peaks at 2 (analytic-only) or 1 (analytic + anti-analytic)
    def get_lp_sum():
        if analytic_only:
            return _compute_lp_sum(psi_fs0, phi_f, J, log2_T)
        return (_compute_lp_sum(psi_fs0, phi_f, J, log2_T) +
                _compute_lp_sum(psi_fs1))

    lp_sum = get_lp_sum()
    if analytic_only:  # time scattering
        for n in range(len(psi_fs0)):
            norm_filter(psi_fs0, peak_idxs, lp_sum, n)
    else:  # frequential scattering
        for s_idx, psi_fs in enumerate(psi_fs_all):
            for n in range(len(psi_fs)):
                norm_filter(psi_fs, peak_idxs[s_idx], lp_sum, n, s_idx)

    if nyquist_correction:
        lp_sum = get_lp_sum()  # compute against latest
        correct_nyquist(psi_fs_all, peak_idxs, lp_sum)

    if passes == 1:
        return scaling_factors
    return energy_norm_filterbank(psi_fs0, psi_fs1, phi_f, J, log2_T,
                                  r_th, passes - 1, scaling_factors)


#### misc / long #############################################################
def conj_fr(x):
    """Conjugate in frequency domain by swapping all bins (except dc);
    assumes frequency along first axis.
    """
    out = np.zeros_like(x)
    out[0] = x[0]
    out[1:] = x[:0:-1]
    return out


def compute_filter_redundancy(p0_f, p1_f):
    """Measures "redundancy" as overlap of energies. Namely, ratio of
    product of energies to mean of energies of Frequency-domain filters
    `p0_f` and `p1_f`.
    """
    p0sq, p1sq = np.abs(p0_f)**2, np.abs(p1_f)**2
    # energy overlap relative to sum of individual energies
    r = np.sum(p0sq * p1sq) / ((p0sq.sum() + p1sq.sum()) / 2)
    return r


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

#### helpers #################################################################
def _compute_lp_sum(psi_fs, phi_f=None, J=None, log2_T=None):
    lp_sum = 0
    for psi_f in psi_fs:
        lp_sum += np.abs(psi_f)**2
    if phi_f is not None and (
            # else lowest frequency bandpasses are too attenuated
            log2_T is not None and J is not None and log2_T >= J):
        lp_sum += np.abs(phi_f)**2
    return lp_sum


def _recalibrate_psi_fr(xi1_frs, sigma1_frs, j1_frs, is_cqt1_frs, N, alpha,
                        N_fr_scales_min, N_fr_scales_max,
                        sigma_max_to_min_max_ratio):
    # recalibrate filterbank to each j0
    # j0=0 is the original length, no change needed
    xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new = (
        {0: xi1_frs}, {0: sigma1_frs}, {0: j1_frs}, {0: is_cqt1_frs})
    scale_diff_max = None

    for N_fr_scales in range(N_fr_scales_max - 1, N_fr_scales_min - 1, -1):
        scale_diff = N_fr_scales_max - N_fr_scales
        (xi1_frs_new[scale_diff], sigma1_frs_new[scale_diff],
         j1_frs_new[scale_diff], is_cqt1_frs_new[scale_diff]) = [], [], [], []
        factor = 2**scale_diff

        # contract largest temporal width of any wavelet by 2**j0,
        # but not above sigma_max/sigma_max_to_min_max_ratio
        sigma_min_max = max(sigma1_frs) / sigma_max_to_min_max_ratio
        new_sigma_min = min(sigma1_frs) * factor
        if new_sigma_min > sigma_min_max:
            scale_diff_max = scale_diff
            break

        # halve distance from existing xi_max to .5 (max possible)
        new_xi_max = .5 - (.5 - max(xi1_frs)) / factor
        new_xi_min = 2 / (N // factor)
        # logarithmically distribute
        new_xi = np.logspace(np.log10(new_xi_min), np.log10(new_xi_max),
                             len(xi1_frs), endpoint=True)[::-1]
        xi1_frs_new[scale_diff].extend(new_xi)
        new_sigma = np.logspace(np.log10(new_sigma_min),
                                np.log10(max(sigma1_frs)),
                                len(xi1_frs), endpoint=True)[::-1]
        sigma1_frs_new[scale_diff].extend(new_sigma)
        for xi, sigma in zip(new_xi, new_sigma):
            j1_frs_new[scale_diff].append(get_max_dyadic_subsampling(
                xi, sigma, alpha=alpha))
            is_cqt1_frs_new[scale_diff].append(False)

    return (xi1_frs_new, sigma1_frs_new, j1_frs_new, is_cqt1_frs_new,
            scale_diff_max)
