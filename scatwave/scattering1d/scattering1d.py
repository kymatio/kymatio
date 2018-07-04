import torch
from torch.autograd import Variable
import numpy as np
import math

from .utils import pad, unpad, real, subsample_fourier, modulus_complex
from .utils import compute_padding, compute_border_indices
from .utils import cast_psi, cast_phi
from .filter_bank import scattering_filter_factory
from .filter_bank import calibrate_scattering_filters
from .fft_wrapper import fft1d_c2c, ifft1d_c2c_normed


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
        Only 'l2' or 'l1' normalizations are supported.
        Defaults to 'l1'
    criterion_amplitude: float >0 and <1, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger criterion_amplitude, the smaller the padding size is.
        Defaults to 1e-3
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent
        wavelets). Defaults to sqrt(0.5).
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering,
        it is equal to sigma0 / 2**J_scattering. Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.
    P_max : int, optional
        maximal number of periods to use to make sure that the
        FFT of the filters is periodic. P_max = 5 is more than enough for
        double precision. Defaults to 5
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to 1e-7

    Returns
    -------
    min_to_pad: int
        minimal value to pad the signal on one size to avoid any
        boundary error
    """
    J_tentative = int(np.ceil(np.log2(T)))
    _, _, _, t_max_phi = scattering_filter_factory(
        J_tentative, J, Q, normalize=normalize, to_torch=False,
        max_subsampling=0, criterion_amplitude=criterion_amplitude,
        r_psi=r_psi, sigma0=sigma0, alpha=alpha, P_max=P_max, eps=eps)
    min_to_pad = 3 * t_max_phi
    return min_to_pad


class Scattering1D(object):
    def __init__(self, T, J, Q, normalize='l1', criterion_amplitude=1e-3,
                 r_psi=math.sqrt(0.5), sigma0=0.1, alpha=5.,
                 P_max=5, eps=1e-7, order2=True, average_U1=True,
                 oversampling=0, vectorize=True):
        """
        Main module implementing the scattering transform in 1D.

        The scattering transform computes two wavelet transform followed
        by modulus non-linearity.
        It can be summarized as:
        S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]
        where
        S_J^0 x = x star phi_J
        S_J^1 x = [|x star psi^1_lambda| star phi_J]_lambda
        S_J^2 x =
            [||x star psi^1_lambda| star psi^2_mu| star phi_J]_{lambda, mu}

        where star denotes the convolution (in time),
        phi_J is a low pass filter, psi^1_lambda is a family of band pass
        filters and psi^2_lambda is another family of band pass filters.

        Only Morlet filters are used in this implementation.

        Convolutions are efficiently performed in the Fourier domain
        with this implementation.

        Example
        -------
        # 1) Define a Scatterer object as:
        s = Scattering1D(T, J, Q)
        #    where T is the temporal size, 2**J the scale of the scattering
        #    and Q the number of intermediate scales per octave
        # N.B.: the design of the filters is optimized for values of Q >= 6
        # 2) (optional) Change its arguments with
        s.set_default_arguments(**kwargs)
        # (especially for second order, etc)
        # 3) Forward on an input Variable x of shape B x 1 x T,
        #     where B is the batch size.
        result_s = s.forward(x)


        Parameters
        ----------
        T : int
            temporal support of the inputs
        J : int
            logscale of the scattering
        Q : int
            number of filters per octave (an integer >= 1)
        normalize : string, optional
            normalization type for the wavelets.
            Only 'l2' or 'l1' normalizations are supported.
            Defaults to 'l1'
        criterion_amplitude: float, optional
            controls the padding size (the larger
            criterion amplitude, the smaller the padding size).
            Measures the amount of the Gaussian mass (in l1) which can be
            ignored after padding. Defaults to 1e-3
        r_psi : float, optional
            Should be >0 and <1. Controls the redundancy of the filters
            (the larger r_psi, the larger the overlap between adjacent
            wavelets). Defaults to sqrt(0.5).
        sigma0 : float, optional
            parameter controlling the frequential width of the
            low-pass filter at J_scattering=0; at a an absolute J_scattering,
            it is equal to sigma0 / 2**J_scattering. Defaults to 1e-1
        alpha : float, optional
            tolerance factor for the aliasing after subsampling.
            The larger alpha, the more conservative the value of maximal
            subsampling is. Defaults to 5.
        P_max : int, optional
            maximal number of periods to use to make sure that the
            FFT of the filters is periodic. P_max = 5 is more than enough for
            double precision. Defaults to 5
        eps : float, optional
            required machine precision for the periodization (single
            floating point is enough for deep learning applications).
            Defaults to 1e-7
        order2 : boolean, optional
            whether to compute the 2nd order scattering or not.
            Defaults to True.
        average_U1 : boolean, optional
            whether to return an averaged first order
            (proper scattering) or simply the |x star psi^1_lambda|.
            Defaults to True.
        oversampling : boolean, optional
            integer >= 0 contrlling the relative
            oversampling relative to the default downsampling by 2**J after
            convolution with phi_J, so that the value 2**(J-oversampling)
            is used. Defaults to 0
        vectorize : boolean, optional
            whether to return a vectorized scattering or
            not (in which case the output is a dictionary).
            Defaults to True.

        Attributes
        ----------
        T : int
            temporal support of the inputs
        J : int
            logscale of the scattering
        Q : int
            number of filters per octave (an integer >= 1)
        J_pad : int
            log2 of the support on which the signal is padded (is a power
            of 2 for efficient FFT implementation)
        pad_left : int
            amount which is padded on the left of the original temporal support
        pad_right : int
            amount which is padded on the right of the original support
        phi_fft : dictionary
            countaining the low-pass filter at all resolutions
            See filters_bank.scat_filters_factory for an exact description
        psi1_fft : dictionary
            Countaining the filters of the 1st order at all
            resolutions. See filters_bank.scat_filters_factory for an exact
            description
        psi1_fft : dictionary
            countaining the filters of the 2nd order at all
            resolutions. See filters_bank.scat_filters_factory for an exact
            description
        default_args : dictionary
            Countains the default arguments with which the scattering should
            be computed
        """
        super(Scattering1D, self).__init__()
        # Store the parameters
        self.T = T
        self.J = J
        self.Q = Q
        self.r_psi = r_psi
        self.sigma0 = sigma0
        self.alpha = alpha
        self.P_max = P_max
        self.eps = eps
        self.criterion_amplitude = criterion_amplitude
        self.normalize = normalize
        # Build internal values
        self.build()
        self.set_default_args(order2=order2, average_U1=average_U1,
                              oversampling=oversampling, vectorize=vectorize)

    def build(self):
        """
        Builds the internal filters.
        """
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
        phi_fft, psi1_fft, psi2_fft, _ = scattering_filter_factory(
            self.J_pad, self.J, self.Q, normalize=self.normalize,
            to_torch=True, criterion_amplitude=self.criterion_amplitude,
            r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha,
            P_max=self.P_max, eps=self.eps)
        self.psi1_fft = psi1_fft
        cast_psi(self.psi1_fft, torch.FloatTensor)
        self.psi2_fft = psi2_fft
        cast_psi(self.psi2_fft, torch.FloatTensor)
        self.phi_fft = phi_fft
        cast_phi(self.phi_fft, torch.FloatTensor)

    def _type(self, target_type):
        cast_psi(self.psi1_fft, target_type)
        cast_psi(self.psi2_fft, target_type)
        cast_phi(self.phi_fft, target_type)
        return self

    def cpu(self):
        """
        Moves the parameters of the scattering to the CPU
        """
        return self._type(torch.FloatTensor)

    def cuda(self):
        """
        Moves the parameters of the scattering to the GPU
        """
        return self._type(torch.cuda.FloatTensor)

    def set_default_args(self, order2=True, average_U1=True, oversampling=0,
                         vectorize=True):
        """
        Allows to dynamically change the type of inputs sent to the module,
        thereby avoiding recomputing the whole filterbank, which is expensive.

        Parameters
        ----------
        order2 : boolean, optional
            whether to include the 2nd order scattering or not.
            Defaults to True.
        average_U1 : boolean, optional
            whether to return an averaged first order
            (proper scattering) or simply the |x star psi^1_lambda|.
            Defaults to True.
        oversampling : boolean, optional
            integer >= 0 contrlling the relative
            oversampling relative to the default downsampling by 2**J after
            convolution with phi_J, so that the value 2**(J-oversampling)
            is used. Defaults to 0
        vectorize : boolean, optional
            whether to return a vectorized scattering or
            not (in which case the output is a dictionary).
            Defaults to True.
        """
        self.default_args = {'order2': order2, 'average_U1': average_U1,
                             'oversampling': oversampling,
                             'vectorize': vectorize}

    def _get_arguments(self, order2, average_U1, oversampling, vectorize):
        new_o2 = self.default_args['order2'] if order2 is None else order2
        new_aU1 = self.default_args['average_U1'] if average_U1 is None\
            else average_U1
        new_os = self.default_args['oversampling'] if oversampling is None\
            else oversampling
        new_vect = self.default_args['vectorize'] if vectorize is None\
            else vectorize
        return new_o2, new_aU1, new_os, new_vect

    def forward(self, x, order2=None, average_U1=None, oversampling=None,
                vectorize=None):
        """
        Forward pass of the scattering.

        It is possible to change the options of the scattering dynamically
        at this point, but we recommend using set_default_args to simplify
        the API when calling the scattering transform.

        If the optional parameters are left to None, then the default value
        set by set_default_args() stored in self.default_args is used.

        Parameters
        ----------
        x : Variable tensor
            torch Variable with 3 dimensions (B, 1, T) where B is arbitrary.
        order2 : boolean, optional
            whether to compute the 2nd order or not. Defaults to None
            (set_default_args() defaults this to True)
        average_U1 : boolean, optional
            whether to average the first modulus wavelet transform or not.
            Defaults to None.
            (set_default_args() defaults this to True)
        oversampling : int, optional
            integer >= 0 contrlling the relative oversampling relative to
            the default downsampling by 2**J after convolution with phi_J,
            so that the value 2**(J-oversampling) is used. Defaults to None.
            (set_default_args() defaults this to 0)
        vectorize: boolean, optional
            whether to return a vectorized scattering or not (in which case
            the output is a dictionary). Defaults to None.
            (set_default_args() defaults this to True)

        Returns
        -------
        S : Variable tensor or dictionary.
            scattering of the input x. If vectorize is True, the output is
            a 3D tensor (B, C, T') such that S[i] = scattering(x[i]).
            If vectorize if False, it is a dictionary with keys
            corresponding to the leafs of the scattering tree.
        """
        # basic checking, should be improved
        if len(x.shape) != 3:
            raise ValueError(
                'Input tensor x should have 3 axis, got {}'.format(
                    len(x.shape)))
        if x.shape[1] != 1:
            raise ValueError(
                'Input tensor should only have 1 channel, got {}'.format(
                    x.shape[1]))
        # get the arguments before calling the scattering
        order2, average_U1, oversampling, vectorize = self._get_arguments(
            order2, average_U1, oversampling, vectorize)
        # treat the arguments
        if vectorize:
            size_scat = self.precompute_size_scattering(
                self.J, self.Q, order2=order2, detail=False)
        else:
            size_scat = 0
        S = scattering(x, self.psi1_fft, self.psi2_fft, self.phi_fft,
                       self.J, order2=order2, average_U1=average_U1,
                       pad_left=self.pad_left, pad_right=self.pad_right,
                       ind_start=self.ind_start, ind_end=self.ind_end,
                       oversampling=oversampling, vectorize=vectorize,
                       size_scat=size_scat)
        return S

    def __call__(self, x):
        return self.forward(x)

    @staticmethod
    def compute_meta_scattering(J, Q, order2=False):
        """
        Computes the meta information on each coordinate of the scattering
        vector.

        Parameters
        ----------
        J : int
            dyadic scale of the scattering transform
        Q : int
            number of wavelets per octave at the first order
        order2 : boolean, optional
            whether the order 2 is used or not. Defaults to False.

        Returns
        -------
        coords : dictionary
            a dictionary whose keys are the coordinates of the scattering
            vector. Each entry is itself a vector with entries
            'order' (to which scattering order corresponds the coordinate),
            'xi': the central frequency of the filter (xi1 and xi2 for 2nd
            order),
            'sigma': the spectral width of the filter (sigma1 and sigma2 for
            second order),
            'key': the key in the dictionaries psi1_fft and psi2_fft (key1 and
            key2 for second order)
        """
        sigma_low, xi1, sigma1, xi2, sigma2 = calibrate_scattering_filters(
            J, Q)
        coords = {}
        coords[0] = {'order': '0', 'sigma': sigma_low, 'xi': 0}
        cc = 1
        for (j1, n1) in sorted(xi1.keys()):
            coords[cc] = {'order': '1', 'xi': xi1[j1, n1],
                          'sigma': sigma1[j1, n1],
                          'key': (j1, n1)}
            cc += 1
            if order2:
                for (j2, n2) in sorted(xi2.keys()):
                    if j2 > j1:
                        coords[cc] = {
                            'order': '2', 'j2': j2,
                            'key1': (j1, n1), 'key2': (j2, n2),
                            'xi2': xi2[j2, n2], 'sigma2': sigma2[j2, n2],
                            'xi1': xi1[j1, n1], 'sigma1': sigma1[j1, n1]}
                        cc += 1
        return coords

    @staticmethod
    def precompute_size_scattering(J, Q, order2=False, detail=False):
        """
        Precomputes the size of the scattering vector.

        Parameters
        ----------
        J : int
            scale of the scattering transform
        Q : int
            number of wavelets per octave for the first order
        order2 : boolean, optional
            whether the 2nd order scattering is also computed.
            Defaults to False
        detail : boolean, optional
            whether to provide a detailed size (order per order) or
            an aggregate

        Returns
        -------
        if detail is True, returns a 2-tuple or 3-tuple size such that
        size[i] = size of order i (with i = 0, 1, 2)
        if detail is False, returns the sum of the above tuple
        """
        sigma_low, xi1, sigma1, xi2, sigma2 = calibrate_scattering_filters(
            J, Q)
        size_order0 = 1
        size_order1 = len(xi1)
        size_order2 = 0
        for (j, n) in sorted(xi1.keys()):
            for (j2, n2) in sorted(xi2.keys()):
                if j2 > j:
                    size_order2 += 1
        if detail:
            return size_order0, size_order1, size_order2
        else:
            if order2:
                return size_order0 + size_order1 + size_order2
            else:
                return size_order0 + size_order1


def scattering(x, psi1, psi2, phi, J, pad_left=0, pad_right=0,
               ind_start=None, ind_end=None, oversampling=0,
               order2=False, average_U1=True, size_scat=0, vectorize=False):
    """
    Main function implementing the scattering computation.

    Parameters
    ----------
    x : Variable tensor
        a torch Variable of size (B, 1, T) where T is the temporal size
    psi1 : dictionary
        a dictionary of filters (in the Fourier domain), with keys (j, n)
        j corresponds to the downsampling factor for x \ast psi1[(j, q)].
        n corresponds to an arbitrary numbering
        * psi1[(j, n)] is itself a dictionary, with keys corresponding to the
        dilation factors: psi1[(j, n)][j2] corresponds to a support of size
        2**(J_max - j2), where J_max has been defined a priori
        (J_max = size of the padding support of the input)
        * psi1[(j, n)] only has real values;
        the tensors are complex so that broadcasting applies
    psi2 : dictionary
        a dictionary of filters, with keys (j2, n2). Same remarks as for psi1
    phi : dictionary
        a dictionary of filters of scale 2^J with keys (j) where j is the
        downsampling factor: phi[j] is a (real) filter
    J : int
        scale of the scattering
    pad_left : int, optional
        how much to pad the signal on the left. Defaults to 0
    pad_right : int, optional
        how much to pad the signal on the right. Defaults to 0
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start
    oversampling : int, optional
        how much to oversample the scattering (with respect to 2**J):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to 0
    order2 : boolean, optional
        Whether to compute the 2nd order or not. Defaults to False.
    average_U1 : boolean, optional
        whether to average the first order vector. Defaults to True
    size_scat : dictionary or int, optional
        contains the number of channels of the scattering,
        precomputed for speed-up. Defaults to 0
    vectorize : boolean, optional
        whether to return a dictionary or a tensor. Defaults to False.
    """
    # S is simply a dictionary if we do not perform the averaging...
    if vectorize:
        batch_size = x.shape[0]
        kJ = max(J - oversampling, 0)
        temporal_size = ind_end[kJ] - ind_start[kJ]
        S = Variable(x.data.new(batch_size, size_scat,
                                temporal_size).fill_(0.))
    else:
        S = {}

    # pad to a dyadic size and make it complex
    U0 = pad(x, pad_left=pad_left, pad_right=pad_right, to_complex=True)
    # compute the FFT
    U0_hat = fft1d_c2c(U0)
    # initialize the cursor
    cc = 0  # current coordinate
    # Get S0
    k0 = max(J - oversampling, 0)
    S0_J_hat = subsample_fourier(U0_hat * phi[0], 2**k0)
    S0_J = unpad(real(ifft1d_c2c_normed(S0_J_hat)),
                 ind_start[k0], ind_end[k0])
    if vectorize:
        S[:, cc] = S0_J
        cc += 1
    else:
        S[0] = S0_J
    # First order:
    for (j, n) in sorted(psi1.keys()):
        # Convolution + downsampling
        k1 = max(j - oversampling, 0)
        assert psi1[(j, n)]['xi'] < 0.5 / (2**k1)
        U1_hat = subsample_fourier(U0_hat * psi1[(j, n)][0], 2**k1)
        # Take the modulus and go back in Fourier
        U1_hat = fft1d_c2c(modulus_complex(ifft1d_c2c_normed(U1_hat)))
        # Convolve with phi_J
        if average_U1:
            k1_J = max(J - k1 - oversampling, 0)
            S1_J_hat = subsample_fourier(U1_hat * phi[k1], 2**k1_J)
            S1_J = unpad(real(ifft1d_c2c_normed(S1_J_hat)),
                         ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            S1_J = U1_hat
        if vectorize:
            S[:, cc] = S1_J
            cc += 1
        else:
            S[j, n] = S1_J
        if order2:
            # 2nd order
            for (j2, n2) in sorted(psi2.keys()):
                if j2 > j:
                    assert psi2[j2, n2]['xi'] < psi1[j, n]['xi']
                    # convolution + downsampling
                    k2 = max(j2 - k1 - oversampling, 0)
                    U2_hat = subsample_fourier(U1_hat * psi2[j2, n2][k1],
                                               2**k2)
                    # take the modulus and go back in Fourier
                    U2_hat = fft1d_c2c(modulus_complex(
                        ifft1d_c2c_normed(U2_hat)))
                    # Convolve with phi_J
                    k2_J = max(J - k2 - k1 - oversampling, 0)
                    S2_J_hat = subsample_fourier(U2_hat * phi[k1 + k2],
                                                 2**k2_J)
                    S2_J = unpad(real(ifft1d_c2c_normed(S2_J_hat)),
                                 ind_start[k1 + k2 + k2_J],
                                 ind_end[k1 + k2 + k2_J])
                    if vectorize:
                        S[:, cc] = S2_J
                        cc += 1
                    else:
                        S[j, n, j2, n2] = S2_J
    return S
