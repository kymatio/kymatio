from ...frontend.base_frontend import ScatteringBase
import math
import numbers

import numpy as np

from ..filter_bank import (scattering_filter_factory, periodize_filter_fourier,
                           psi_fr_factory, phi_fr_factory,
                           morlet_1d, gauss_1d)
from ..utils import (compute_border_indices, compute_padding,
                     compute_minimum_support_to_pad,
                     compute_meta_scattering,
                     compute_meta_jtfs,
                     precompute_size_scattering)


class ScatteringBase1D(ScatteringBase):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, T=None, vectorize=True, out_type='array',
            pad_mode='reflect', max_pad_factor=2, backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q if isinstance(Q, tuple) else (Q, 1)
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.T = T
        self.vectorize = vectorize
        self.out_type = out_type
        self.pad_mode = pad_mode
        self.max_pad_factor = max_pad_factor
        self.backend = backend

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
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3
        self.normalize = 'l1'

        # check the shape
        if isinstance(self.shape, numbers.Integral):
            self.N = self.shape
        elif isinstance(self.shape, tuple):
            self.N = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")

        # check pad_mode
        if self.pad_mode not in ('reflect', 'zero'):
            raise ValueError("`pad_mode` must be one of: reflect, zero "
                             "(got %s)" % self.pad_mode)

        # ensure 2**J <= nextpow2(N)
        mx = 2**math.ceil(math.log2(self.N))
        if 2**(self.J) > mx:
            raise ValueError(("2**J cannot exceed input length (rounded up to "
                              "pow2) (got {} > {})".format(2**(self.J), mx)))

        # check T or set default
        if self.T is None:
            self.T = 2**(self.J)
        elif self.T == 'global':
            self.T == mx
        elif self.T > self.N:
            raise ValueError("The temporal support T of the low-pass filter "
                             "cannot exceed input length (got {} > {})".format(
                                 self.T, self.N))
        self.log2_T = math.floor(math.log2(self.T))
        self.average_global = bool(self.T == mx)

        # Compute the minimum support to pad (ideally)
        min_to_pad = compute_minimum_support_to_pad(
            self.N, self.J, self.Q, self.T, r_psi=self.r_psi,
            sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude,
            normalize=self.normalize, pad_mode=self.pad_mode)

        J_pad = int(np.ceil(np.log2(self.N + 2 * min_to_pad)))
        if self.max_pad_factor is None:
            self.J_pad = J_pad
        else:
            J_max_support = int(round(np.log2(self.N * 2**self.max_pad_factor)))
            self.J_pad = min(J_pad, J_max_support)

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.pad_left, 2**self.J_pad - self.pad_right)

        # check that we get any second-order coefficients if max_order==2
        meta = ScatteringBase1D.meta(self)
        if self.max_order == 2 and np.isnan(meta['n'][-1][1]):
            raise ValueError("configuration yields no second-order coefficients; "
                             "try increasing `J`, or (`Scattering1D` only) "
                             "set `max_order=1`.")

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f, _ = scattering_filter_factory(
            self.J_pad, self.J, self.Q, self.T,
            normalize=self.normalize,
            criterion_amplitude=self.criterion_amplitude, r_psi=self.r_psi,
            sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return compute_meta_scattering(self.J, self.Q, self.J_pad, self.T,
                                       max_order=self.max_order)

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

        return precompute_size_scattering(
            self.J, self.Q, max_order=self.max_order, detail=detail)

    _doc_shape = 'N'

    _doc_instantiation_shape = {True: 'S = Scattering1D(J, N, Q)',
                                False: 'S = Scattering1D(J, Q)'}

    _doc_param_shape = \
    r"""shape : int
            The length of the input signals.
        """

    _doc_attrs_shape = \
    r"""J_pad : int
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
            creation. Defaults to `True`.
        """

    _doc_attr_average = \
    r"""average : boolean
            Controls whether the output should be averaged (the standard
            scattering transform) or not (resulting in wavelet modulus
            coefficients). Note that to obtain unaveraged output, the
            `vectorize` flag must be set to `False` or `out_type` must be set
            to `'list'`.
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
        {param_shape}Q : int
            The number of first-order wavelets per octave.
        Q2 : int
            The number of second-order wavelets per octave.
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


class TimeFrequencyScatteringBase1D():
    def __init__(self, J_fr=None, Q_fr=1, F=None, average_fr=True,
                 oversampling_fr=0, aligned=True, resample_filters_fr=True,
                 pad_mode='zero'):
        self._J_fr = J_fr
        self._Q_fr = Q_fr
        self._F = F
        self._average_fr = average_fr
        self.oversampling_fr = oversampling_fr
        self.aligned = aligned
        if isinstance(resample_filters_fr, tuple):
            self.resample_psi_fr, self.resample_phi_fr = resample_filters_fr
        else:
            self.resample_psi_fr = self.resample_phi_fr = resample_filters_fr
        self.pad_mode = pad_mode

    def build(self):
        # don't allow untested and unneeded combination
        if not self.aligned and self.out_type == "list":
            raise ValueError("`aligned=False` is only allowed with "
                             "`out_type` = 'array' or 'array-like'")

        self._shape_fr = self.get_shape_fr()
        max_order_fr = 1
        if self._J_fr is None:  # TODO set from shape_fr_max
            self._J_fr = int(math.log2(self.Q[0]) + 1)
        # number of psi1 filters
        self._n_psi1 = len(self.psi1_f)

        self.sc_freq = _FrequencyScatteringBase(
            self._J_fr, self._shape_fr, self._Q_fr, self._F, max_order_fr,
            self._average_fr, self.resample_psi_fr, self.resample_phi_fr,
            self.oversampling, self.vectorize, self.out_type, self.pad_mode,
            self._n_psi1, self.backend)
        self.finish_creating_filters()

    def get_shape_fr(self):
        """This is equivalent to `len(x)` along frequency, which varies across
        `psi2`, so we compute for each.
        """
        shape_fr = []
        for n2 in range(len(self.psi2_f)):
            j2 = self.psi2_f[n2]['j']
            max_freq_nrows = 0
            for n1 in range(len(self.psi1_f)):
                if j2 > self.psi1_f[n1]['j']:
                    max_freq_nrows += 1
            shape_fr.append(max_freq_nrows)
        return shape_fr

    def finish_creating_filters(self):
        """Handles necessary adjustments in time scattering filters unaccounted
        for in default construction.
        """
        # ensure phi is subsampled up to (log2_T - 1) for `phi_t * psi_f` pairs
        max_sub_phi = lambda: max(k for k in self.phi_f if isinstance(k, int))
        while max_sub_phi() < self.log2_T:
            self.phi_f[max_sub_phi() + 1] = periodize_filter_fourier(
                self.phi_f[0], nperiods=2**(max_sub_phi() + 1))

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_jtfs()` with the parameters of the
        transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_jtfs()`.
        """
        return compute_meta_jtfs(self.J, self.Q, self.J_pad, self.J_pad_fr_max,
                                 self.T, self.F, self.J_fr, self.Q_fr)

    # access key attributes via frequential class
    @property
    def J_fr(self):
        return self.sc_freq.J_fr

    @property
    def Q_fr(self):
        return self.sc_freq.Q_fr

    @property
    def shape_fr_max(self):
        return self.sc_freq.shape_fr_max

    @property
    def J_pad_fr_max(self):
        return self.sc_freq.J_pad_max

    @property
    def average_fr(self):
        return self.sc_freq.average

    @property
    def average_fr_global(self):
        return self.sc_freq.average_global

    @property
    def F(self):
        return self.sc_freq.F

    @property
    def log2_F(self):
        return self.sc_freq.log2_F

    @property
    def max_order_fr(self):
        return self.sc_freq.max_order_fr

    @classmethod
    def _document(cls):
        # TODO documentation
        cls.__doc__ = """Joint time-frequency scattering"""


class _FrequencyScatteringBase(ScatteringBase):
    """Attribute object for TimeFrequencyScatteringBase1D for frequential
    scattering part of JTFS.
    """
    def __init__(self, J_fr, shape_fr, Q_fr=2, F=None, max_order_fr=1,
                 average=True, resample_psi_fr=True, resample_phi_fr=True,
                 oversampling=0, vectorize=True, out_type='array',
                 pad_mode='zero', n_psi1=None, backend=None):
        super(_FrequencyScatteringBase, self).__init__()
        self.J_fr = J_fr
        self.shape_fr = shape_fr
        self.Q_fr = Q_fr
        self.F = F
        self.max_order_fr = max_order_fr
        self.average = average
        self.resample_psi_fr = resample_psi_fr
        self.resample_phi_fr = resample_phi_fr
        self.oversampling = oversampling
        self.vectorize = vectorize
        self.out_type = out_type
        self._n_psi1 = n_psi1
        self.pad_mode = pad_mode
        self.backend = backend

        self.build()
        self.create_phi_filters()
        self.compute_padding_fr()
        self.create_psi_filters()

    def build(self):
        """""" # TODO
        self.r_psi = math.sqrt(0.5)
        self.sigma0 = 0.1
        self.alpha = 5.
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3
        self.normalize = 'l1'

        # longest obtainable frequency row w.r.t. which we calibrate filters
        self.shape_fr_max = max(self.shape_fr)

        # ensure 2**J_fr <= nextpow2(shape_fr_max)
        mx = 2**math.ceil(math.log2(self.shape_fr_max))
        if 2**(self.J_fr) > mx:
            raise ValueError(("2**J_fr cannot exceed maximum number of frequency "
                              "rows (rounded up to pow2) in joint scattering "
                              "(got {} > {})".format(2**(self.J_fr), mx)))

        # check F or set default
        if self.F is None:
            self.F = 2**(self.J_fr)
        elif self.F == 'global':
            self.F = mx
        elif self.F > mx:
            raise ValueError("The temporal support F of the low-pass filter "
                             "cannot exceed maximum number of frequency rows "
                             "(rounded up to pow2) in joint scattering "
                             "(got {} > {})".format(self.F, mx))
        self.log2_F = math.floor(math.log2(self.F))
        self.average_global = bool(self.F == mx)

        # compute maximum amount of padding
        self.J_pad_max, self.min_to_pad_max = self._compute_J_pad(
            self.shape_fr_max, (self.Q_fr, 0))

    def create_phi_filters(self):
        self.phi_f = phi_fr_factory(
            self.F, self.log2_F, self.Q_fr, self.J_pad_max,
             **self.get_params('resample_phi_fr', 'criterion_amplitude',
                               'sigma0', 'P_max', 'eps'))

        if self.resample_phi_fr and not self.average_global:
            # subsampling before `_joint_lowpass()` (namely `* sc_freq.phi_f`)
            # is limited by `sc_freq.phi_f[0]`'s time width.
            # This is accounted for  in `scattering_filter_factory_fr` by
            # not computing `sc_freq.phi_f` at such resampling lengths.
            n_phi_f = max(k for k in self.phi_f if isinstance(k, int))
            self.max_subsampling_before_phi_fr = n_phi_f
        else:
            # usual behavior
            self.max_subsampling_before_phi_fr = self.log2_F

        # Edge case: need phi_f in case number of `psi1` filters (and thus
        # first-order coeffs) pads greater than max(J_pad). Note this filter,
        # if `!= phi_f[0]`, always computes per `resample_phi_fr=True`
        # TODO compute per `resample_phi_fr` instead?
        # Q=(0,0) to decide from `phi` alone, since we don't do *psi here
        self.J_pad_fo = self.compute_J_pad(self._n_psi1, recompute=True, Q=(0, 0))
        diff = self.J_pad_fo - self.J_pad_max
        if diff > 0:
            self.phi_f_fo = gauss_1d(2**self.J_pad_fo, sigma=self.phi_f['sigma'],
                                     P_max=self.P_max, eps=self.eps)
            # need to subsample more since padded more
            self.log2_F_fo = self.log2_F + 1
        else:
            self.phi_f_fo = self.phi_f[-diff]
        # subsample more or less since padded more (_fo > _max) or less (< _max)
        self.log2_F_fo = self.log2_F + diff


    def create_psi_filters(self):
        self.psi1_f_up, self.psi1_f_down = psi_fr_factory(
            self.J_fr, self.Q_fr, self.J_pad_max, self.j0s,
            **self.get_params('resample_psi_fr', 'r_psi', 'normalize',
                              'sigma0', 'alpha', 'P_max', 'eps'))

        # create `_fo` filters for `phi_t * psi_f` pairs
        # TODO compute per `resample_psi_fr` instead?
        def copy_meta(out, p):
            out.update({k: v for k, v in p.items() if not isinstance(k, int)})

        diff = self.J_pad_fo - self.J_pad_max
        def create_psi_fo(p):
            out = {}
            if diff > 0 or (diff < 0 and self.resample_psi_fr):
                out[0] = morlet_1d(2**self.J_pad_fo, p['xi'], p['sigma'],
                                   **self.get_params('normalize', 'P_max', 'eps'))
            elif diff < 0:  # TODO
                out[0] = periodize_filter_fourier(p, nperiods=2**(-diff))
            else:
                out[0] = p[0].copy()
            copy_meta(out, p)
            return out

        self.psi1_f_up_fo = [create_psi_fo(p) for p in self.psi1_f_up]
        self.psi1_f_down_fo = [create_psi_fo(p) for p in self.psi1_f_down]

    def compute_padding_fr(self):
        """Long story"""  # TODO
        attrs = ('J_pad', 'pad_left', 'pad_right', 'ind_start', 'ind_end', 'j0s')
        for attr in attrs:
            setattr(self, attr, [])

        # J_pad is ordered lower to greater, so iterate backward then reverse
        # (since we don't yet know max `j0`)
        pad_prev = -1
        for shape_fr in self.shape_fr[::-1]:
            if shape_fr != 0:
                J_pad = self.compute_J_pad(shape_fr)

                # compute the padding quantities
                pad_left = 0
                pad_right = 2**J_pad - pad_left - shape_fr
                if pad_prev == -1:
                    j0 = 0
                elif J_pad < pad_prev:
                    j0 += 1
                pad_prev = J_pad

                # compute unpad indices for all possible subsamplings
                ind_start, ind_end = [], []
                for j in range(self.log2_F + 1):
                    if j == j0:
                        ind_start.append(0)
                        ind_end.append(shape_fr)
                    elif j > j0:
                        ind_start.append(0)
                        ind_end.append(math.ceil(ind_end[-1] / 2))
                    else:
                        ind_start.append(-1)
                        ind_end.append(-1)
            else:
                J_pad, pad_left, pad_right, j0 = -1, -1, -1, -1
                ind_start, ind_end = [], []

            self.J_pad.append(J_pad)
            self.pad_left.append(pad_left)
            self.pad_right.append(pad_right)
            self.ind_start.append(ind_start)
            self.ind_end.append(ind_end)
            self.j0s.append(j0)

        for attr in attrs:
            getattr(self, attr).reverse()

        # compute maximum ind_start and ind_end across all subsampling factors
        # to use as common for out_type='array'
        def get_idxs(attr):
            return getattr(self, attr.strip('_max'))

        for attr in ('ind_start_max', 'ind_end_max'):
            setattr(self, attr, [])
            for j in range(self.log2_F + 1):
                idxs_max = max(get_idxs(attr)[n2][j]
                               for n2 in range(len(self.shape_fr))
                               if len(get_idxs(attr)[n2]) != 0)
                getattr(self, attr).append(idxs_max)

    def compute_J_pad(self, shape_fr, recompute=False, Q=(0, 0)):
        """Depends on `shape_fr` and `(resample_phi_fr or resample_phi_fr)`:
            True:  pad per `shape_fr` and `min_to_pad` of longest `shape_fr`
            False: pad per `shape_fr` and `min_to_pad` of subsampled filters

        `min_to_pad` is computed for both `phi_f[0]` and `psi1_f[-1]` in case
        latter has greater time-domain support.

        `recompute=True` will force computation from `shape_fr` alone, independent
        of `J_pad_max` and `min_to_pad_max`, and per `resample_* == True`.
        """
        if recompute:
            J_pad, _ = self._compute_J_pad(shape_fr, Q)

        elif self.resample_phi_fr or self.resample_psi_fr:
            J_pad = math.ceil(np.log2(shape_fr + 2 * self.min_to_pad_max))
        else:
            # reproduce `compute_minimum_support_to_pad`'s logic
            # TODO instead compute directly? must introduce kwargs to above method
            J_tentative     = int(np.ceil(np.log2(shape_fr)))
            J_tentative_max = int(np.ceil(np.log2(self.shape_fr_max)))
            J_pad = self.J_pad_max - (J_tentative_max - J_tentative)

        # don't let J_pad drop below `J_pad_max - max_sub...`
        J_pad = max(J_pad,
                    self.J_pad_max - self.max_subsampling_before_phi_fr)
        return J_pad

    def _compute_J_pad(self, shape_fr, Q):
        min_to_pad = compute_minimum_support_to_pad(
            shape_fr, self.J_fr, Q, self.F,
            **self.get_params('r_psi', 'sigma0', 'alpha', 'P_max', 'eps',
                              'criterion_amplitude', 'normalize', 'pad_mode'))
        J_pad = math.ceil(np.log2(shape_fr + 2 * min_to_pad))
        return J_pad, min_to_pad

    def get_params(self, *args):
        return {k: getattr(self, k) for k in args}


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase1D']
