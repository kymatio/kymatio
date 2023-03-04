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
    low_pass_f = low_pass_f.reshape(2 * P - 1, -1).mean(axis=0)
    if xi:
        # define the gabor at freq xi and the low-pass, both of width sigma
        gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
        # discretize in signal <=> periodize in Fourier
        gabor_f = gabor_f.reshape(2 * P - 1, -1).mean(axis=0)
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


def get_max_dyadic_subsampling(xi, sigma, alpha, **unused_kwargs):
    """
    Computes the maximal dyadic subsampling which is possible for a Gabor
    filter of frequency xi and width sigma

    Finds the maximal integer j such that:
    omega_0 < 2^{-(j + 1)}
    where omega_0 is the boundary of the filter, defined as
    omega_0 = abs(xi) + alpha * sigma

    This ensures that the filter can be subsampled by a factor 2^j without
    aliasing.

    We use the same formula for Gabor and Morlet filters.

    Parameters
    ----------
    xi : float
        frequency of the filter in [-0.5, 0.5]
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
    upper_bound = min(abs(xi) + alpha * sigma, 0.5)
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


def anden_generator(J, Q, sigma0, r_psi, **unused_kwargs):
    """
    Yields the center frequencies and bandwidths of a filterbank, in compliance
    with the ScatNet package. Center frequencies follow a geometric progression
    of common factor 2**(1/Q) above a certain "elbow frequency" xi_elbow and
    an arithmetic progression of common difference (1/Q) below xi_elbow.

    The corresponding bandwidth sigma is proportional to center frequencies
    for xi>=xi_elbow and are constant (sigma=sigma_min) for xi<xi_elbow.

    The formula for xi_elbow is quite complicated and involves four hyperparameters
    J, Q, r_psi, and sigma0:

    xi_elbow = compute_xi_max(Q) * (sigma0/2**J)/compute_sigma_psi(xi, Q, r_psi)

    where compute_xi_max and compute_sigma_psi are defined elsewhere in this module.

    Intuitively, the role of xi_elbow is to make the filterbank as "wavelet-like"
    as possible (common xi/sigma ratio) while guaranteeing a lower bound on sigma
    (hence an upper bound on time support) and full coverage of the Fourier
    domain between pi/2**J and pi.

    Parameters
    ----------
    J : int
        log-scale of the scattering transform, such that wavelets of both
        filterbanks have a maximal support that is proportional to 2**J.

    Q : int
        number of wavelets per octave in the geometric progression portion of
        the filterbank.

    r_psi : float in (0, 1)
        Should be >0 and <1. The higher the r_psi, the greater the sigmas.
        Adjacent wavelets peak at 1 and meet at r_psi.

    sigma0 : float
        Should be >0. The minimum bandwidth is sigma0/2**J.
    """
    xi = compute_xi_max(Q)
    sigma = compute_sigma_psi(xi, Q, r=r_psi)
    sigma_min = sigma0 / 2**J

    if sigma <= sigma_min:
        xi = sigma
    else:
        yield xi, sigma
        # High-frequency (constant-Q) region: geometric progression of xi
        while sigma > (sigma_min * math.pow(2, 1 / Q)):
            xi /= math.pow(2, 1 / Q)
            sigma /= math.pow(2, 1 / Q)
            yield xi, sigma

    # Low-frequency (constant-bandwidth) region: arithmetic progression of xi
    elbow_xi = xi
    for q in range(Q-1):
        xi -= 1/Q * elbow_xi
        yield xi, sigma_min


def spin(filterbank_fn, filterbank_kwargs):
    def spinned_fn(J, Q, **kwargs):
        yield from filterbank_fn(J, Q, **kwargs)
        for xi, sigma in filterbank_fn(J, Q, **kwargs):
            yield -xi, sigma
    return spinned_fn, filterbank_kwargs


def scattering_filter_factory(N, J, Q, T, filterbank, _reduction=np.sum):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': normalized center frequency, where 0.5 corresponds to Nyquist.
    * 'sigma': normalized bandwidth in the Fourier.
    * 'j': log2 of downsampling factor after filtering. j=0 means no downsampling,
        j=1 means downsampling by one half, etc.
    * 'levels': list of NumPy arrays containing the filter at various levels
        of downsampling. levels[0] is at full resolution, levels[1] at half
        resolution, etc.

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
    filterbank : tuple (callable filterbank_fn, dict filterbank_kwargs)
        filterbank_fn should take J and Q as positional arguments and
        **filterbank_kwargs as optional keyword arguments.
        Corresponds to the self.filterbank property of the scattering object.
        As of v0.3, only anden_generator is supported as filterbank_fn.
    _reduction : callable
        either np.sum (default) or np.mean.

    Returns
    -------
    phi_f, psi1_f, psi2_f ... : dictionaries
        phi_f corresponds to the low-pass filter and psi1_f, psi2_f, to the
        wavelet filterbanks at layers 1 and 2 respectively.
        See above for a description of the dictionary structure.
    """
    filterbank_fn, filterbank_kwargs = filterbank
    max_j = 0
    previous_J = 0
    log2_T = math.floor(math.log2(T))
    psis_f = []

    # Loop over scattering layers
    for Q_layer in Q:
        psi_f = []

        # Loop over center frequencies xi and bandwidths sigma in the filterbank
        for xi, sigma in filterbank_fn(J, Q_layer, **filterbank_kwargs):

            # Construct filter at full resolution
            psi_levels = [morlet_1d(N, xi, sigma)]
            j = get_max_dyadic_subsampling(xi, sigma, **filterbank_kwargs)

            # Resample to smaller resolutions if necessary (beyond 1st layer)
            # The idiom min(previous_J, j, 1+log2_T) implements "j1 < j2"
            for level in range(1, min(previous_J, j, 1+log2_T)):
                psi_level_reshaped = psi_levels[0].reshape(2**level, -1)
                psi_level = _reduction(psi_level_reshaped, axis=0)
                psi_levels.append(psi_level)
            psi_f.append({'levels': psi_levels, 'xi': xi, 'sigma': sigma, 'j': j})
            max_j = max(j, max_j)

        # Keep score of how many resolutions will be needed in the next layer
        previous_J = max_j
        psis_f.append(psi_f)

    # Same with low-pass filter phi
    sigma_low = filterbank_kwargs["sigma0"] / T
    phi_levels = [gauss_1d(N, sigma_low)]
    for level in range(1, max(previous_J, 1+log2_T)):
        phi_level_reshaped = phi_levels[0].reshape(2**level, -1)
        phi_level = _reduction(phi_level_reshaped, axis=0)
        phi_levels.append(phi_level)
    phi_f = {'levels': phi_levels, 'xi': 0, 'sigma': sigma_low, 'j': log2_T,
        'N': N}

    return tuple([phi_f] + psis_f)
