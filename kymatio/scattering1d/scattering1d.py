# Authors: Mathieu Andreux, Joakim Andén, Edouard Oyallon
# Scientific Ancestry: Joakim Andén, Mathieu Andreux, Vincent Lostanlen

import math
import numbers
import numpy as np
import torch

from .backend import (fft1d_c2c, ifft1d_c2c, modulus_complex, pad, real,
    subsample_fourier, unpad)
from .filter_bank import (calibrate_scattering_filters,
    scattering_filter_factory)
from .utils import cast_phi, cast_psi, compute_border_indices, compute_padding


__all__ = ['Scattering1D']


class Scattering1D(object):
    """The 1D scattering transform

    The scattering transform computes a cascade of wavelet transforms
    alternated with a complex modulus non-linearity. The scattering transform
    of a 1D signal :math:`x(t)` may be written as

        $S_J x = [S_J^{(0)} x, S_J^{(1)} x, S_J^{(2)} x]$

    where

        $S_J^{(0)} x(t) = x \\star \\phi_J(t)$,

        $S_J^{(1)} x(t, \\lambda) =|x \\star \\psi_\\lambda^{(1)}| \\star \\phi_J$, and

        $S_J^{(2)} x(t, \\lambda, \\mu) = |\\,| x \\star \\psi_\\lambda^{(1)}| \\star \\psi_\\mu^{(2)} | \\star \\phi_J$.

    In the above formulas, :math:`\\star` denotes convolution in time. The
    filters $\\psi_\\lambda^{(1)}(t)$ and $\\psi_\\mu^{(2)}(t)$
    are analytic wavelets with center frequencies $\\lambda$ and
    $\\mu$, while $\\phi_J(t)$ is a real lowpass filter centered
    at the zero frequency.

    The `Scattering1D` class implements the 1D scattering transform for a
    given set of filters whose parameters are specified at initialization.
    While the wavelets are fixed, other parameters may be changed after the
    object is created, such as whether to compute all of :math:`S_J^{(0)} x`,
    $S_J^{(1)} x$, and $S_J^{(2)} x$ or just $S_J^{(0)} x$
    and $S_J^{(1)} x$.

    The scattering transform may be computed on the CPU (the default) or a
    GPU, if available. A `Scattering1D` object may be transferred from one
    to the other using the `cuda()` and `cpu()` methods.

    Given an input Tensor `x` of size `(B, T)`, where `B` is the number of
    signals to transform (the batch size) and `T` is the length of the signal,
    we compute its scattering transform by passing it to the `forward()`
    method.

    Example
    -------
    ::

        # Set the parameters of the scattering transform.
        J = 6
        T = 2**13
        Q = 8

        # Generate a sample signal.
        x = torch.randn(1, 1, T)

        # Define a Scattering1D object.
        S = Scattering1D(J, T, Q)

        # Calculate the scattering transform.
        Sx = S.forward(x)

    Above, the length of the signal is `T = 2**13 = 8192`, while the maximum
    scale of the scattering transform is set to `2**J = 2**6 = 64`. The
    time-frequency resolution of the first-order wavelets
    :math:`\\psi_\\lambda^{(1)}(t)` is set to `Q = 8` wavelets per octave.
    The second-order wavelets :math:`\\psi_\\mu^{(2)}(t)` always have one
    wavelet per octave.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform. In other words,
        the maximum scale is given by `2**J`.
    T : int
        The length of the input signals.
    Q : int >= 1
        The number of first-order wavelets per octave (second-order wavelets
        are fixed to one wavelet per octave). Defaults to `1`.
    max_order : int, optional
        The maximum order of scattering coefficients to compute. Must be either
        `1` or `2`. Defaults to `2`.
    average : boolean, optional
        Determines whether the output is averaged in time or not. The averaged
        output corresponds to the standard scattering transform, while the
        un-averaged output skips the last convolution by :math:`\\phi_J(t)`.
        This parameter may be modified after object creation.
        Defaults to `True`.
    oversampling : integer >= 0, optional
        Controls the oversampling factor relative to the default as a power
        of two. Since the convolving by wavelets (or lowpass filters) and
        taking the modulus reduces the high-frequency content of the signal,
        we can subsample to save space and improve performance. However, this
        may reduce precision in the calculation. If this is not desirable,
        `oversampling` can be set to a large value to prevent too much
        subsampling. This parameter may be modified after object creation.
        Defaults to `0`.
    vectorize : boolean, optional
        Determines wheter to return a vectorized scattering transform (that
        is, a large array containing the output) or a dictionary (where each
        entry corresponds to a separate scattering coefficient). This parameter
        may be modified after object creation. Defaults to True.

    Attributes
    ----------
    J : int
        The maximum log-scale of the scattering transform. In other words,
        the maximum scale is given by `2**J`.
    shape : int
        The length of the input signals.
    Q : int
        The number of first-order wavelets per octave (second-order wavelets
        are fixed to one wavelet per octave).
    J_pad : int
        The logarithm of the padded length of the signals.
    pad_left : int
        The amount of padding to the left of the signal.
    pad_right : int
        The amount of padding to the right of the signal.
    phi_f : dictionary
        A dictionary containing the lowpass filter at all resolutions. See
        `filter_bank.scattering_filter_factory` for an exact description.
    psi1_f : dictionary
        A dictionary containing all the first-order wavelet filters, each
        represented as a dictionary containing that filter at all
        resolutions. See `filter_bank.scattering_filter_factory` for an exact
        description.
    psi2_f : dictionary
        A dictionary containing all the second-order wavelet filters, each
        represented as a dictionary containing that filter at all
        resolutions. See `filter_bank.scattering_filter_factory` for an exact
        description.
        description
    max_order : int
        The maximum scattering order of the transform.
    average : boolean
        Controls whether the output should be averaged (the standard
        scattering transform) or not (resulting in wavelet modulus
        coefficients). Note that to obtain unaveraged output, the `vectorize`
        flag must be set to `False`.
    oversampling : int
        The number of powers of two to oversample the output compared to the
        default subsampling rate determined from the filters.
    vectorize : boolean
        Controls whether the output should be vectorized into a single Tensor
        or collected into a dictionary. For more details, see the
        documentation for `forward()`.
    """
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
                 oversampling=0, vectorize=True):
        super(Scattering1D, self).__init__()
        # Store the parameters
        self.J = J
        self.shape = shape
        self.Q = Q

        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.vectorize = vectorize

        # Build internal values
        self.build()

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """

        # Set these default values for now. In the future, we'll want some
        # flexibility for these, but for now, let's keep them fixed.
        self.r_psi = math.sqrt(0.5)
        self.sigma0 = 0.1
        self.alpha = 5.
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3
        self.normalize = 'l1'

        # check the shape
        if isinstance(self.shape, numbers.Integral):
            self.T = self.shape
        elif isinstance(self.shape, tuple):
            self.T = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")

        # Compute the minimum support to pad (ideally)
        min_to_pad = compute_minimum_support_to_pad(
            self.T, self.J, self.Q, r_psi=self.r_psi, sigma0=self.sigma0,
            alpha=self.alpha, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude,
            normalize=self.normalize)
        # to avoid padding more than T - 1 on the left and on the right,
        # since otherwise torch sends nans
        J_max_support = int(np.floor(np.log2(3 * self.T - 2)))
        self.J_pad = min(int(np.ceil(np.log2(self.T + 2 * min_to_pad))),
                         J_max_support)
        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.T)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.J, self.pad_left, self.pad_left + self.T)
        # Finally, precompute the filters
        phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(
            self.J_pad, self.J, self.Q, normalize=self.normalize,
            to_torch=True, criterion_amplitude=self.criterion_amplitude,
            r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha,
            P_max=self.P_max, eps=self.eps)
        self.psi1_f = psi1_f
        self.psi2_f = psi2_f
        self.phi_f = phi_f
        self._type(torch.FloatTensor)

    def _type(self, target_type):
        """Change the datatype of the filters

        This function is used internally to convert the filters. It does not
        need to be called explicitly.

        Parameters
        ----------
        target_type : type
            The desired type of the filters, typically `torch.FloatTensor`
            or `torch.cuda.FloatTensor`.
        """
        cast_psi(self.psi1_f, target_type)
        cast_psi(self.psi2_f, target_type)
        cast_phi(self.phi_f, target_type)
        return self

    def cpu(self):
        """Move to the CPU

        This function prepares the object to accept input Tensors on the CPU.
        """
        return self._type(torch.FloatTensor)

    def cuda(self):
        """Move to the GPU

        This function prepares the object to accept input Tensors on the GPU.
        """
        return self._type(torch.cuda.FloatTensor)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return Scattering1D.compute_meta_scattering(
            self.J, self.Q, max_order=self.max_order)

    def output_size(self, detail=False):
        """Get size of the scattering transform

        Calls the static method `precompute_size_scattering()` with the
        parameters of the transform object.

        Parameters
        ----------
        detail : boolean, optional
            Specifies whether to provide a detailed size (number of coefficient
            per order) or an aggregate size (total number of coefficients).

        Returns
        ------
        size : int or tuple
            See the documentation for `precompute_size_scattering()`.
        """

        return Scattering1D.precompute_size_scattering(
            self.J, self.Q, max_order=self.max_order, detail=detail)

    def forward(self, x):
        """Apply the scattering transform

        Given an input Tensor of size `(B, T0)`, where `B` is the batch
        size and `T0` is the length of the individual signals, this function
        computes its scattering transform. If the `vectorize` flag is set to
        `True`, the output is in the form of a Tensor or size `(B, C, T1)`,
        where `T1` is the signal length after subsampling to the scale `2**J`
        (with the appropriate oversampling factor to reduce aliasing), and
        `C` is the number of scattering coefficients.  If `vectorize` is set
        `False`, however, the output is a dictionary containing `C` keys, each
        a tuple whose length corresponds to the scattering order and whose
        elements are the sequence of filter indices used.

        Furthermore, if the `average` flag is set to `False`, these outputs
        are not averaged, but are simply the wavelet modulus coefficients of
        the filters.

        Parameters
        ----------
        x : tensor
            An input Tensor of size `(B, T0)`.

        Returns
        -------
        S : tensor or dictionary
            If the `vectorize` flag is `True`, the output is a Tensor
            containing the scattering coefficients, while if `vectorize`
            is `False`, it is a dictionary indexed by tuples of filter indices.
        """
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            if not(self.average):
                raise ValueError(
                    'Options average=False and vectorize=True are ' +
                    'mutually incompatible. Please set vectorize to False.')
            size_scattering = self.precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0
        S = scattering(x, self.psi1_f, self.psi2_f, self.phi_f,
                       self.J, max_order=self.max_order, average=self.average,
                       pad_left=self.pad_left, pad_right=self.pad_right,
                       ind_start=self.ind_start, ind_end=self.ind_end,
                       oversampling=self.oversampling,
                       vectorize=self.vectorize,
                       size_scattering=size_scattering)

        if self.vectorize:
            scattering_shape = S.shape[-2:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            for k, v in S.items():
                scattering_shape = v.shape[-2:]
                S[k] = v.reshape(batch_shape + scattering_shape)

        return S

    def __call__(self, x):
        return self.forward(x)

    @staticmethod
    def compute_meta_scattering(J, Q, max_order=2):
        """Get metadata on the transform.

        This information specifies the content of each scattering coefficient,
        which order, which frequencies, which filters were used, and so on.

        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform.
            In other words, the maximum scale is given by `2**J`.
        Q : int >= 1
            The number of first-order wavelets per octave.
            Second-order wavelets are fixed to one wavelet per octave.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`. Defaults to `2`.

        Returns
        -------
        meta : dictionary
            A dictionary with the following keys:

            - `'order`' : tensor
                A Tensor of length `C`, the total number of scattering
                coefficients, specifying the scattering order.
            - `'xi'` : tensor
                A Tensor of size `(C, max_order)`, specifying the center
                frequency of the filter used at each order (padded with NaNs).
            - `'sigma'` : tensor
                A Tensor of size `(C, max_order)`, specifying the frequency
                bandwidth of the filter used at each order (padded with NaNs).
            - `'j'` : tensor
                A Tensor of size `(C, max_order)`, specifying the dyadic scale
                of the filter used at each order (padded with NaNs).
            - `'n'` : tensor
                A Tensor of size `(C, max_order)`, specifying the indices of
                the filters used at each order (padded with NaNs).
            - `'key'` : list
                The tuples indexing the corresponding scattering coefficient
                in the non-vectorized output.
        """
        sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = \
            calibrate_scattering_filters(J, Q)

        meta = {}

        meta['order'] = [[], [], []]
        meta['xi'] = [[], [], []]
        meta['sigma'] = [[], [], []]
        meta['j'] = [[], [], []]
        meta['n'] = [[], [], []]
        meta['key'] = [[], [], []]

        meta['order'][0].append(0)
        meta['xi'][0].append(())
        meta['sigma'][0].append(())
        meta['j'][0].append(())
        meta['n'][0].append(())
        meta['key'][0].append(())

        for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
            meta['order'][1].append(1)
            meta['xi'][1].append((xi1,))
            meta['sigma'][1].append((sigma1,))
            meta['j'][1].append((j1,))
            meta['n'][1].append((n1,))
            meta['key'][1].append((n1,))

            if max_order < 2:
                continue

            for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
                if j2 > j1:
                    meta['order'][2].append(2)
                    meta['xi'][2].append((xi1, xi2))
                    meta['sigma'][2].append((sigma1, sigma2))
                    meta['j'][2].append((j1, j2))
                    meta['n'][2].append((n1, n2))
                    meta['key'][2].append((n1, n2))

        for field, value in meta.items():
            meta[field] = value[0] + value[1] + value[2]

        pad_fields = ['xi', 'sigma', 'j', 'n']
        pad_len = max_order

        for field in pad_fields:
            meta[field] = [x+(math.nan,)*(pad_len-len(x)) for x in meta[field]]

        array_fields = ['order', 'xi', 'sigma', 'j', 'n']

        for field in array_fields:
            meta[field] = torch.from_numpy(np.array(meta[field]))

        return meta

    @staticmethod
    def precompute_size_scattering(J, Q, max_order=2, detail=False):
        """Get size of the scattering transform

        The number of scattering coefficients depends on the filter
        configuration and so can be calculated using a few of the scattering
        transform parameters.

        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform.
            In other words, the maximum scale is given by `2**J`.
        Q : int >= 1
            The number of first-order wavelets per octave.
            Second-order wavelets are fixed to one wavelet per octave.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`. Defaults to `2`.
        detail : boolean, optional
            Specifies whether to provide a detailed size (number of coefficient
            per order) or an aggregate size (total number of coefficients).

        Returns
        -------
        size : int or tuple
            If `detail` is `False`, returns the number of coefficients as an
            integer. If `True`, returns a tuple of size `max_order` containing
            the number of coefficients in each order.
        """
        sigma_low, xi1, sigma1, j1, xi2, sigma2, j2 = \
            calibrate_scattering_filters(J, Q)

        size_order0 = 1
        size_order1 = len(xi1)
        size_order2 = 0
        for n1 in range(len(xi1)):
            for n2 in range(len(xi2)):
                if j2[n2] > j1[n1]:
                    size_order2 += 1
        if detail:
            return size_order0, size_order1, size_order2
        else:
            if max_order == 2:
                return size_order0 + size_order1 + size_order2
            else:
                return size_order0 + size_order1


def compute_minimum_support_to_pad(T, J, Q, criterion_amplitude=1e-3,
                                   normalize='l1', r_psi=math.sqrt(0.5),
                                   sigma0=1e-1, alpha=5., P_max=5, eps=1e-7):
    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    T : int
        temporal size of the input signal
    J : int
        scale of the scattering
    Q : int
        number of wavelets per octave
    normalize : string, optional
        normalization type for the wavelets.
        Only `'l2'` or `'l1'` normalizations are supported.
        Defaults to `'l1'`
    criterion_amplitude: float `>0` and `<1`, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger criterion_amplitude, the smaller the padding size is.
        Defaults to `1e-3`
    r_psi : float, optional
        Should be `>0` and `<1`. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent
        wavelets).
        Defaults to `sqrt(0.5)`.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering,
        it is equal to :math:`\\frac{\\sigma_0}{2^J}`.
        Defaults to `1e-1`.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger the alpha, the more conservative the value of maximal
        subsampling is.
        Defaults to `5`.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic.
        `P_max = 5` is more than enough for double precision.
        Defaults to `5`.
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to `1e-7`.

    Returns
    -------
    min_to_pad: int
        minimal value to pad the signal on one size to avoid any
        boundary error.
    """
    J_tentative = int(np.ceil(np.log2(T)))
    _, _, _, t_max_phi = scattering_filter_factory(
        J_tentative, J, Q, normalize=normalize, to_torch=False,
        max_subsampling=0, criterion_amplitude=criterion_amplitude,
        r_psi=r_psi, sigma0=sigma0, alpha=alpha, P_max=P_max, eps=eps)
    min_to_pad = 3 * t_max_phi
    return min_to_pad


def scattering(x, psi1, psi2, phi, J, pad_left=0, pad_right=0,
               ind_start=None, ind_end=None, oversampling=0,
               max_order=2, average=True, size_scattering=(0, 0, 0), vectorize=False):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    x : Tensor
        a torch Tensor of size `(B, 1, T)` where `T` is the temporal size
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
    pad_left : int, optional
        how much to pad the signal on the left. Defaults to `0`
    pad_right : int, optional
        how much to pad the signal on the right. Defaults to `0`
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start
    oversampling : int, optional
        how much to oversample the scattering (with respect to :math:`2^J`):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to `0`
    order2 : boolean, optional
        Whether to compute the 2nd order or not. Defaults to `False`.
    average_U1 : boolean, optional
        whether to average the first order vector. Defaults to `True`
    size_scattering : tuple
        Contains the number of channels of the scattering, precomputed for
        speed-up. Defaults to `(0, 0, 0)`.
    vectorize : boolean, optional
        whether to return a dictionary or a tensor. Defaults to False.

    """
    # S is simply a dictionary if we do not perform the averaging...
    if vectorize:
        batch_size = x.shape[0]
        kJ = max(J - oversampling, 0)
        temporal_size = ind_end[kJ] - ind_start[kJ]
        S = x.new(batch_size, sum(size_scattering), temporal_size).fill_(0.)
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
        S[:, cc[0], :] = S0_J.squeeze(dim=1)
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
            S[:, cc[1], :] = S1_J.squeeze(dim=1)
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
                        S[:, cc[2], :] = S2_J.squeeze(dim=1)
                        cc[2] += 1
                    else:
                        S[n1, n2] = S2_J

    return S
