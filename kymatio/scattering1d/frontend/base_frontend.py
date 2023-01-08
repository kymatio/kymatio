from ...frontend.base_frontend import ScatteringBase
from collections import Counter
import math
import numbers
import numpy as np
from warnings import warn

from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering import (joint_timefrequency_scattering,
    jtfs_average_and_format)
from ..filter_bank import (compute_temporal_support, gauss_1d,
    anden_generator, scattering_filter_factory, spin)
from ..utils import compute_border_indices, compute_padding, parse_T


class ScatteringBase1D(ScatteringBase):
    def __init__(self, J, shape, Q=1, T=None, max_order=2,
                 oversampling=0, out_type='array', backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.oversampling = oversampling
        self.out_type = out_type
        self.backend = backend
        self._reduction = np.mean

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        self.r_psi = math.sqrt(0.5)
        self.sigma0 = 0.1
        self.alpha = 5.

        # check the number of filters per octave
        if np.any(np.array(self.Q) < 1):
            raise ValueError('Q must always be >= 1, got {}'.format(self.Q))

        if isinstance(self.Q, int):
            self.Q = (self.Q, 1)
        elif isinstance(self.Q, tuple):
            if len(self.Q) == 1:
                self.Q = self.Q + (1, )
            elif len(self.Q) < 1 or len(self.Q) > 2:
                raise NotImplementedError("Q must be an integer, 1-tuple or "
                                          "2-tuple. Scattering transforms "
                                          "beyond order 2 are not implemented.")
        else:
            raise ValueError("Q must be an integer or a tuple")

        # check input length
        if isinstance(self.shape, numbers.Integral):
            self.shape = (self.shape,)
        elif isinstance(self.shape, tuple):
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")
        N_input = self.shape[0]

        # check T or set default
        self.T, self.average = parse_T(self.T, self.J, N_input)
        self.log2_T = math.floor(math.log2(self.T))

        # Compute the minimum support to pad (ideally)
        phi_f = gauss_1d(N_input, self.sigma0/self.T)
        min_to_pad = 3 * compute_temporal_support(
            phi_f.reshape(1, -1), criterion_amplitude=1e-3)

        # to avoid padding more than N - 1 on the left and on the right,
        # since otherwise torch sends nans
        J_max_support = int(np.floor(np.log2(3 * N_input - 2)))
        J_pad = min(int(np.ceil(np.log2(N_input + 2 * min_to_pad))),
                    J_max_support)
        self._N_padded = 2**J_pad

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self._N_padded, N_input)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, self.pad_left + N_input)

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(
            self._N_padded, self.J, self.Q, self.T, self.filterbank, self._reduction)
        ScatteringBase._check_filterbanks(self.psi1_f, self.psi2_f)

    def scattering(self, x):
        ScatteringBase1D._check_runtime_args(self)
        ScatteringBase1D._check_input(self, x)

        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)

        U_0 = self.backend.pad(x, pad_left=self.pad_left, pad_right=self.pad_right)

        filters = [self.phi_f, self.psi1_f, self.psi2_f][:(1+self.max_order)]
        S_gen = scattering1d(U_0, self.backend, filters,
            self.oversampling, (self.average=='local'))

        if self.out_type in ['array', 'list']:
            S = list()
        elif self.out_type == 'dict':
            S = dict()

        for path in S_gen:
            path['order'] = len(path['n'])
            if self.average == 'local':
                res = max(self.log2_T - self.oversampling, 0)
            elif path['order']>0:
                res = max(path['j'][-1] - self.oversampling, 0)
            else:
                res = 0

            if self.average == 'global':
                path['coef'] = self.backend.average_global(path['coef'])
            else:
                path['coef'] = self.backend.unpad(
                    path['coef'], self.ind_start[res], self.ind_end[res])
            path['coef'] = self.backend.reshape_output(
                path['coef'], batch_shape, n_kept_dims=1)

            if self.out_type in ['array', 'list']:
                S.append(path)
            elif self.out_type == 'dict':
                S[path['n']] = path['coef']

        if self.out_type == 'dict':
            return S

        S.sort(key=(lambda path: (path['order'], path['n'])))

        if self.out_type == 'array':
            S = self.backend.concatenate([path['coef'] for path in S], dim=-2)

        return S

    def meta(self):
        """Get metadata on the transform.

        This information specifies the content of each scattering coefficient,
        which order, which frequencies, which filters were used, and so on.

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
        backend = self._DryBackend()
        filters = [self.phi_f, self.psi1_f, self.psi2_f][:(1+self.max_order)]
        S_gen = scattering1d(
            None, backend, filters, self.oversampling, average_local=False)
        S = sorted(list(S_gen), key=lambda path: (len(path['n']), path['n']))
        meta = dict(order=np.array([len(path['n']) for path in S]))
        meta['key'] = [path['n'] for path in S]
        meta['n'] = np.stack([np.append(
            path['n'], (np.nan,)*(self.max_order-len(path['n']))) for path in S])
        filterbanks = (self.psi1_f, self.psi2_f)[:self.max_order]
        for key in ['xi', 'sigma', 'j']:
            meta[key] = meta['n'] * np.nan
            for order, filterbank in enumerate(filterbanks):
                for n, psi in enumerate(filterbank):
                    meta[key][meta['n'][:, order]==n, order] = psi[key]
        return meta

    def output_size(self, detail=False):
        """Number of scattering coefficients.

        Parameters
        ----------
        detail : boolean, optional
            Whether to aggregate the count (detail=False, default) across
            orders or to break it down by scattering depth (layers 0, 1, and 2).

        Returns
        ------
        size : int or tuple
            If `detail=False` (default), total number of scattering coefficients.
            Else, number of coefficients at zeroth, first, and second order.
        """
        if detail:
            return tuple(Counter(self.meta()['order']).values())
        return len(self.meta()['key'])

    def _check_runtime_args(self):
        if not self.out_type in ('array', 'dict', 'list'):
            raise ValueError("out_type must be one of 'array', 'dict'"
                             ", or 'list'. Got: {}".format(self.out_type))

        if not self.average and self.out_type == 'array':
            raise ValueError("Cannot convert to out_type='array' with "
                             "T=0. Please set out_type to 'dict' or 'list'.")

        if self.oversampling < 0:
            raise ValueError("oversampling must be nonnegative. Got: {}".format(
                self.oversampling))

        if not isinstance(self.oversampling, numbers.Integral):
            raise ValueError("oversampling must be integer. Got: {}".format(
                self.oversampling))

    def _check_input(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

    @property
    def J_pad(self):
        warn("The attribute J_pad is deprecated and will be removed in v0.4. "
        "Measure len(self.phi_f[0]) for the padded length (previously 2**J_pad) "
        "or access shape[0] for the unpadded length (previously N).", DeprecationWarning)
        return int(np.log2(self._N_padded))

    @property
    def N(self):
        warn("The attribute N is deprecated and will be removed in v0.4. "
        "Measure len(self.phi_f[0]) for the padded length (previously 2**J_pad) "
        "or access shape[0] for the unpadded length (previously N).", DeprecationWarning)
        return int(self.shape[0])

    @property
    def filterbank(self):
        filterbank_kwargs = {
            "alpha": self.alpha, "r_psi": self.r_psi, "sigma0": self.sigma0}
        return (anden_generator, filterbank_kwargs)

    _doc_shape = 'N'

    _doc_instantiation_shape = {True: 'S = Scattering1D(J, N, Q)',
                                False: 'S = Scattering1D(J, Q)'}

    _doc_param_shape = \
    r"""shape : int
            The length of the input signals.
        """

    _doc_attrs_shape = \
    r"""pad_left : int
            The amount of padding to the left of the signal.
        pad_right : int
            The amount of padding to the right of the signal.
        phi_f : dictionary
            A dictionary containing the lowpass filter at all resolutions. See
            `filter_bank.scattering_filter_factory` for an exact description.
        psi1_f : dictionary
            A dictionary containing all the first-order wavelet filters, each
            represented as a dictionary containing that filter at all
            resolutions. See `filter_bank.scattering_filter_factory` for an
            exact description.
        psi2_f : dictionary
            A dictionary containing all the second-order wavelet filters, each
            represented as a dictionary containing that filter at all
            resolutions. See `filter_bank.scattering_filter_factory` for an
            exact description.
        """

    _doc_param_average = \
    r"""average : boolean, optional
            Determines whether the output is averaged in time or not. The
            averaged output corresponds to the standard scattering transform,
            while the un-averaged output skips the last convolution by
            :math:`\phi_J(t)`.  This parameter may be modified after object
            creation. Defaults to `True`. Deprecated in v0.3 in favour of `T`
            and will  be removed in v0.4. Replace `average=False` by `T=0` and
            set `T>1` or leave `T=None` for `average=True` (default).
        """

    _doc_attr_average = \
    r"""average : boolean
            Controls whether the output should be averaged (the standard
            scattering transform) or not (resulting in wavelet modulus
            coefficients). Note that to obtain unaveraged output, the
            `vectorize` flag must be set to `False` or `out_type` must be set
            to `'list'`. Deprecated in favor of `T`. For more details,
            see the documentation for `scattering`.
     """

    _doc_param_vectorize = \
    r"""vectorize : boolean, optional
            Determines wheter to return a vectorized scattering transform
            (that is, a large array containing the output) or a dictionary
            (where each entry corresponds to a separate scattering
            coefficient). This parameter may be modified after object
            creation. Deprecated in favor of `out_type` (see below). Defaults
            to True.
        out_type : str, optional
            The format of the output of a scattering transform. If set to
            `'list'`, then the output is a list containing each individual
            scattering coefficient with meta information. Otherwise, if set to
            `'array'`, the output is a large array containing the
            concatenation of all scattering coefficients. Defaults to
            `'array'`.
        """

    _doc_attr_vectorize = \
    r"""vectorize : boolean
            Controls whether the output should be vectorized into a single
            Tensor or collected into a dictionary. Deprecated in favor of
            `out_type`. For more details, see the documentation for
            `scattering`.
        out_type : str
            Specifices the output format of the transform, which is currently
            one of `'array'` or `'list`'. If `'array'`, the output is a large
            array containing the scattering coefficients. If `'list`', the
            output is a list of dictionaries, each containing a scattering
            coefficient along with meta information. For more information, see
            the documentation for `scattering`.
        """

    _doc_class = \
    r"""The 1D scattering transform

        The scattering transform computes a cascade of wavelet transforms
        alternated with a complex modulus non-linearity. The scattering
        transform of a 1D signal :math:`x(t)` may be written as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x(t) = x \star \phi_J(t)$,

            $S_J^{{(1)}} x(t, \lambda) = |x \star \psi_\lambda^{{(1)}}| \star \phi_J$, and

            $S_J^{{(2)}} x(t, \lambda, \mu) = |\,| x \star \psi_\lambda^{{(1)}}| \star \psi_\mu^{{(2)}} | \star \phi_J$.

        In the above formulas, :math:`\star` denotes convolution in time. The
        filters $\psi_\lambda^{{(1)}}(t)$ and $\psi_\mu^{{(2)}}(t)$ are analytic
        wavelets with center frequencies $\lambda$ and $\mu$, while
        $\phi_J(t)$ is a real lowpass filter centered at the zero frequency.

        The `Scattering1D` class implements the 1D scattering transform for a
        given set of filters whose parameters are specified at initialization.
        While the wavelets are fixed, other parameters may be changed after
        the object is created, such as whether to compute all of
        :math:`S_J^{{(0)}} x`, $S_J^{{(1)}} x$, and $S_J^{{(2)}} x$ or just
        $S_J^{{(0)}} x$ and $S_J^{{(1)}} x$.
        {frontend_paragraph}
        Given an input `{array}` `x` of shape `(B, N)`, where `B` is the
        number of signals to transform (the batch size) and `N` is the length
        of the signal, we compute its scattering transform by passing it to
        the `scattering` method (or calling the alias `{alias_name}`). Note
        that `B` can be one, in which case it may be omitted, giving an input
        of shape `(N,)`.

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 6
            N = 2 ** 13
            Q = 8

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering1D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while the
        maximum scale of the scattering transform is set to :math:`2^J = 2^6 =
        64`. The time-frequency resolution of the first-order wavelets
        :math:`\psi_\lambda^{{(1)}}(t)` is set to `Q = 8` wavelets per octave.
        The second-order wavelets :math:`\psi_\mu^{{(2)}}(t)` always have one
        wavelet per octave.

        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform. In other words,
            the maximum scale is given by :math:`2^J`.
        {param_shape}Q : int or tuple
            By default, Q (int) is the number of wavelets per octave for the first
            order and that for the second order has one wavelet per octave. This
            default value can be modified by passing Q as a tuple with two values,
            i.e. Q = (Q1, Q2), where Q1 and Q2 are the number of wavelets per
            octave for the first and second order, respectively.
        T : int
            temporal support of low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        {param_average}oversampling : integer >= 0, optional
            Controls the oversampling factor relative to the default as a
            power of two. Since the convolving by wavelets (or lowpass
            filters) and taking the modulus reduces the high-frequency content
            of the signal, we can subsample to save space and improve
            performance. However, this may reduce precision in the
            calculation. If this is not desirable, `oversampling` can be set
            to a large value to prevent too much subsampling. This parameter
            may be modified after object creation. Defaults to `0`.
        {param_vectorize}
        Attributes
        ----------
        J : int
            The maximum log-scale of the scattering transform. In other words,
            the maximum scale is given by `2 ** J`.
        {param_shape}Q : int
            The number of first-order wavelets per octave (second-order
            wavelets are fixed to one wavelet per octave).
        T : int
            temporal support of low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling
        {attrs_shape}max_order : int
            The maximum scattering order of the transform.
        {attr_average}oversampling : int
            The number of powers of two to oversample the output compared to
            the default subsampling rate determined from the filters.
        {attr_vectorize}"""

    _doc_scattering = \
    """Apply the scattering transform

       Given an input `{array}` of size `(B, N)`, where `B` is the batch
       size (it can be potentially an integer or a shape) and `N` is the length
       of the individual signals, this function computes its scattering
       transform. If the `vectorize` flag is set to `True` (or if it is not
       available in this frontend), the output is in the form of a `{array}`
       or size `(B, C, N1)`, where `N1` is the signal length after subsampling
       to the scale :math:`2^J` (with the appropriate oversampling factor to
       reduce aliasing), and `C` is the number of scattering coefficients. If
       `vectorize` is set `False`, however, the output is a dictionary
       containing `C` keys, each a tuple whose length corresponds to the
       scattering order and whose elements are the sequence of filter indices
       used.

       Note that the `vectorize` flag has been deprecated in favor of the
       `out_type` parameter. If this is set to `'array'` (the default), the
       `vectorize` flag is still respected, but if not, `out_type` takes
       precedence. The two current output types are `'array'` and `'list'`.
       The former gives the type of output described above. If set to
       `'list'`, however, the output is a list of dictionaries, each
       dictionary corresponding to a scattering coefficient and its associated
       meta information. The coefficient is stored under the `'coef'` key,
       while other keys contain additional information, such as `'j'` (the
       scale of the filter used) and `'n`' (the filter index).

       Furthermore, if the `average` flag is set to `False`, these outputs
       are not averaged, but are simply the wavelet modulus coefficients of
       the filters.

       Parameters
       ----------
       x : {array}
           An input `{array}` of size `(B, N)`.

       Returns
       -------
       S : tensor or dictionary
           If `out_type` is `'array'` and the `vectorize` flag is `True`, the
           output is a{n} `{array}` containing the scattering coefficients,
           while if `vectorize` is `False`, it is a dictionary indexed by
           tuples of filter indices. If `out_type` is `'list'`, the output is
           a list of dictionaries as described above.
    """

    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''

        param_average = cls._doc_param_average if cls._doc_has_out_type else ''
        attr_average = cls._doc_attr_average if cls._doc_has_out_type else ''
        param_vectorize = cls._doc_param_vectorize if cls._doc_has_out_type else ''
        attr_vectorize = cls._doc_attr_vectorize if cls._doc_has_out_type else ''

        cls.__doc__ = ScatteringBase1D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call,
            instantiation=instantiation,
            param_shape=param_shape,
            attrs_shape=attrs_shape,
            param_average=param_average,
            attr_average=attr_average,
            param_vectorize=param_vectorize,
            attr_vectorize=attr_vectorize,
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        cls.scattering.__doc__ = ScatteringBase1D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


class TimeFrequencyScatteringBase(ScatteringBase1D):
    def __init__(self, *, J, J_fr, shape, Q, T=None, oversampling=0,
            Q_fr=1, F=None, oversampling_fr=0,
            out_type='array', format='joint', backend=None):
        max_order = 2
        super(TimeFrequencyScatteringBase, self).__init__(J, shape, Q, T,
            max_order, oversampling, out_type, backend)
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.oversampling_fr = oversampling_fr
        self.format = format
        self._reduction = np.sum

    def build(self):
        super(TimeFrequencyScatteringBase, self).build()
        super(TimeFrequencyScatteringBase, self).create_filters()

        # check the number of filters per octave
        if np.any(np.array(self.Q_fr) < 1):
            raise ValueError('Q_fr must be >= 1, got {}'.format(self.Q_fr))

        if isinstance(self.Q_fr, int):
            self.Q_fr = (self.Q_fr,)
        elif isinstance(self.Q_fr, tuple):
            if (len(self.Q_fr) != 1):
                raise NotImplementedError("Q_fr must be an integer or 1-tuple. "
                                          "Time-frequency scattering "
                                          "beyond order 2 is not implemented.")
        else:
            raise ValueError("Q_fr must be an integer or 1-tuple.")

        # check F or set default
        N_input_fr = len(self.psi1_f)
        self.F, self.average_fr = parse_T(
            self.F, self.J_fr, N_input_fr, T_alias='F')
        self.log2_F = math.floor(math.log2(self.F))

        # Compute the minimum support to pad (ideally)
        min_to_pad_fr = 8 * min(self.F, 2 ** self.J_fr)

        # We want to pad the frequency domain to the minimum number that is:
        # (1) greater than number of first-order coefficients, N_input_fr,
        #     by a margin of at least min_to_pad_fr
        # (2) a multiple of all subsampling factors of frequential scattering:
        #     2**1, 2**2, etc. up to 2**K_fr = (2**J_fr / 2**oversampling_fr)
        K_fr = max(self.J_fr - self.oversampling_fr, 0)
        N_padded_fr_subsampled = (N_input_fr + min_to_pad_fr) // (2 ** K_fr)
        self._N_padded_fr = N_padded_fr_subsampled * (2 ** K_fr)

    def create_filters(self):
        phi0_fr_f,= scattering_filter_factory(self._N_padded_fr,
            self.J_fr, (), self.F, self.filterbank_fr, self._reduction)
        phi1_fr_f, psis_fr_f = scattering_filter_factory(self._N_padded_fr,
            self.J_fr, self.Q_fr, 2**self.J_fr, self.filterbank_fr,
            self._reduction)
        self.filters_fr = (phi0_fr_f, [phi1_fr_f] + psis_fr_f)

        # Check for absence of aliasing
        assert all((abs(psi1["xi"]) < 0.5/(2**psi1["j"])) for psi1 in psis_fr_f)

    def scattering(self, x):
        TimeFrequencyScatteringBase._check_runtime_args(self)
        TimeFrequencyScatteringBase._check_input(self, x)

        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)
        U_0 = self.backend.pad(
            x, pad_left=self.pad_left, pad_right=self.pad_right)

        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        U_gen = joint_timefrequency_scattering(U_0, self.backend,
            filters, self.oversampling, (self.average=='local'),
            self.filters_fr, self.oversampling_fr, (self.average_fr=='local'))

        S_gen = jtfs_average_and_format(U_gen, self.backend,
            self.phi_f, self.oversampling, self.average,
            self.filters_fr[0], self.oversampling_fr, self.average_fr,
            self.out_type, self.format)

        # Zeroth order
        path = next(S_gen)
        if not self.average == 'global':
            res = self.log2_T if self.average else 0
            path['coef'] = self.backend.unpad(
                path['coef'], self.ind_start[res], self.ind_end[res])
        path['coef'] = self.backend.reshape_output(
            path['coef'], batch_shape, n_kept_dims=1)
        S = [path]

        # First and second order
        for path in S_gen:
            # Temporal unpadding. Switch cases:
            # 1. If averaging is global, no need for unpadding at all.
            # 2. If averaging is local, unpad at resolution log2_T
            # 3. If there is no averaging, unpadding depends on order:
            #     3a. at order 1, unpad Y_1_fr at resolution log2_T
            #     3b. at order 2, unpad Y_2_fr at resolution j2
            # (for simplicity, we assume oversampling=0 in the rationale above,
            #  but the implementation below works for any value of oversampling)
            if not self.average == 'global':
                if not self.average and len(path['n']) > 1:
                    # Case 3b.
                    res = max(path['j'][-1] - self.oversampling, 0)
                else:
                    # Cases 2a, 2b, and 3a.
                    res = max(self.log2_T - self.oversampling, 0)
                # Cases 2a, 2b, 3a, and 3b.
                path['coef'] = self.backend.unpad(
                    path['coef'], self.ind_start[res], self.ind_end[res])

            # Reshape path to batch shape.
            path['coef'] = self.backend.reshape_output(path['coef'],
                batch_shape, n_kept_dims=(1 + (self.format == "joint")))
            S.append(path)

        if (self.format == 'joint') and (self.out_type == 'array'):
            # Skip zeroth order
            S = S[1:]
            # Concatenate first and second orders into a 4D tensor:
            # (batch, n_jtfs, freq, time) where n_jtfs aggregates (n2, n_fr)
            return self.backend.concatenate([path['coef'] for path in S], dim=-3)
        elif (self.format == "time") and (self.out_type == "array"):
            # Concatenate zeroth, first, and second orders into a 3D tensor:
            # (batch, n_jtfs, time) where n_jtfs aggregates (n1, n2, n_fr)
            return self.backend.concatenate([path['coef'] for path in S], dim=-2)
        elif self.out_type == 'dict':
            return {path['n']: path['coef'] for path in S}
        elif self.out_type == 'list':
            return S

    def meta(self):
        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        U_gen = joint_timefrequency_scattering(None, self._DryBackend(),
            filters, self.oversampling, self.average=='local',
            self.filters_fr, self.oversampling_fr, self.average_fr=='local')
        S_gen = jtfs_average_and_format(U_gen, self._DryBackend(),
            self.phi_f, self.oversampling, self.average,
            self.filters_fr[0], self.oversampling_fr, self.average_fr,
            self.out_type, self.format)
        S = sorted(list(S_gen), key=lambda path: (len(path['n']), path['n']))
        meta = dict(key=[path['n'] for path in S], n=[], n_fr=[], order=[])
        for path in S:
            if len(path['n']) == 0:
                # If format='joint' and out_type='array' skip zeroth order
                if not (self.format == 'joint' and self.out_type == 'array'):
                    # Zeroth order: no n1, no n_fr, no n2
                    meta['n'].append([np.nan, np.nan])
                    meta['n_fr'].append(np.nan)
                    meta['order'].append(0)
            else:
                if len(path['n']) == 1:
                    # First order and format='joint': n=(n_fr,)
                    n1_range = range(0, path['n1_max'], path['n1_stride'])
                    meta['n'].append([n1_range, np.nan])
                elif len(path['n']) == 2 and self.format == 'joint':
                    # Second order and format='joint': n=(n2, n_fr)
                    n1_range = range(0, path['n1_max'], path['n1_stride'])
                    meta['n'].append([n1_range, path['n'][0]])
                elif len(path['n']) == 2 and self.format == 'time':
                    # First order and format='time': n=(n1, n_fr)
                    meta['n'].append([path['n'][0], np.nan])
                elif len(path['n']) == 3 and self.format == 'time':
                    # Second order and format='time': n=(n1, n2, n_fr)
                    meta['n'].append(path['n'][:2])
                meta['n_fr'].append(path['n_fr'][0])
                meta['order'].append(len(path['n']) - (self.format == 'time'))
        meta['n'] = np.array(meta['n'], dtype=object)
        meta['n_fr'] = np.array(meta['n_fr'])
        meta['order'] = np.array(meta['order'])
        for key in ['xi', 'sigma', 'j']:
            meta[key] = np.zeros((meta['n_fr'].shape[0], 2)) * np.nan
            for order, filterbank in enumerate(filters[1:]):
                for n, psi in enumerate(filterbank):
                    meta[key][meta['n'][:, order]==n, order] = psi[key]
            meta[key + '_fr'] = meta['n_fr'] * np.nan
            for n_fr, psi_fr in enumerate(self.filters_fr[1]):
                meta[key + '_fr'][meta['n_fr']==n_fr] = psi_fr[key]
        meta['spin'] = np.sign(meta['xi_fr'])
        return meta

    def _check_runtime_args(self):
        super(TimeFrequencyScatteringBase, self)._check_runtime_args()

        if self.format == 'joint':
            if (not self.average_fr) and (self.out_type == 'array'):
                raise ValueError("Cannot convert to format='joint' with "
                "out_type='array' and F=0. Either set format='time', "
                "out_type='dict', or out_type='list'.")

        if self.oversampling_fr < 0:
            raise ValueError("oversampling_fr must be nonnegative. Got: {}".format(
                self.oversampling_fr))

        if not isinstance(self.oversampling_fr, numbers.Integral):
            raise ValueError("oversampling_fr must be integer. Got: {}".format(
                self.oversampling_fr))

        if self.format not in ['time', 'joint']:
            raise ValueError("format must be 'time' or 'joint'. Got: {}".format(
                self.format))

    @property
    def filterbank_fr(self):
        filterbank_kwargs = {
            "alpha": self.alpha, "r_psi": self.r_psi, "sigma0": self.sigma0}
        return spin(anden_generator, filterbank_kwargs)


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase']
