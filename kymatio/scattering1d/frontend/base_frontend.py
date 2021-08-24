from ...frontend.base_frontend import ScatteringBase
import math
import numbers
import warnings
from types import FunctionType

import numpy as np

from ..filter_bank import (scattering_filter_factory, periodize_filter_fourier,
                           psi_fr_factory, phi_fr_factory,
                           energy_norm_filterbank_tm, energy_norm_filterbank_fr)
from ..utils import (compute_border_indices, compute_padding,
                     compute_minimum_support_to_pad,
                     compute_meta_scattering,
                     compute_meta_jtfs,
                     precompute_size_scattering)


class ScatteringBase1D(ScatteringBase):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, out_type='array', pad_mode='reflect',
            max_pad_factor=2, analytic=False, r_psi=math.sqrt(.5), backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q if isinstance(Q, tuple) else (Q, 1)
        self.T = T
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.out_type = out_type
        self.pad_mode = pad_mode
        self.max_pad_factor = max_pad_factor
        self.analytic = analytic
        self.r_psi = r_psi if isinstance(r_psi, tuple) else (r_psi, r_psi)
        self.backend = backend

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        self.sigma0 = 0.1
        self.alpha = 4.
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
        # dyadic scale of N, also min possible padding
        self.N_scale = math.ceil(math.log2(self.N))

        # check `pad_mode`, set `pad_fn`
        if isinstance(self.pad_mode, FunctionType):
            def pad_fn(x):
                return self.pad_mode(x, self.pad_left, self.pad_right)
            self.pad_mode = 'custom'
        elif self.pad_mode not in ('reflect', 'zero'):
            raise ValueError(("unsupported `pad_mode` '{}';\nmust be a "
                              "function, or string, one of: 'zero', 'reflect'."
                              ).format(str(self.pad_mode)))
        else:
            def pad_fn(x):
                return self.backend.pad(x, self.pad_left, self.pad_right,
                                        self.pad_mode)
        self.pad_fn = pad_fn

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
        self.average_global_phi = bool(self.T == mx)
        self.average_global = bool(self.average_global_phi and self.average)

        # Compute the minimum support to pad (ideally)
        min_to_pad, pad_phi, pad_psi1, pad_psi2 = compute_minimum_support_to_pad(
            self.N, self.J, self.Q, self.T, r_psi=self.r_psi,
            sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude,
            normalize=self.normalize, pad_mode=self.pad_mode)
        if self.average_global:
            min_to_pad = max(pad_psi1, pad_psi2)  # ignore phi's padding

        J_pad = math.ceil(math.log2(self.N + 2 * min_to_pad))
        if self.max_pad_factor is None:
            self.J_pad = J_pad
        else:
            self.J_pad = min(J_pad, self.N_scale + self.max_pad_factor)

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, 2**self.J_pad - self.pad_right)

        # record whether configuration yields second order filters
        meta = ScatteringBase1D.meta(self)
        self._no_second_order_filters = (self.max_order < 2 or
                                         bool(np.isnan(meta['n'][-1][1])))

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(
            self.J_pad, self.J, self.Q, self.T,
            normalize=self.normalize,
            criterion_amplitude=self.criterion_amplitude, r_psi=self.r_psi,
            sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps)

        # energy norm
        energy_norm_filterbank_tm(self.psi1_f, self.psi2_f, self.phi_f,
                                  self.J, self.log2_T)
        # analyticity
        if self.analytic:
          for psi_fs in (self.psi1_f, self.psi2_f):
            for p in psi_fs:
              for k in p:
                if isinstance(k, int):
                    M = len(p[k])
                    p[k][M//2 + 1:] = 0  # zero negatives
                    p[k][M//2] /= 2      # halve Nyquist

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
            self.J, self.Q, self.T, max_order=self.max_order, detail=detail)

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
        pad_mode : str (default 'reflect') / function, optional
            Name of padding scheme to use, one of (`x = [1, 2, 3]`):
                - zero:    [0, 0, 1, 2, 3, 0, 0, 0]
                - reflect: [3, 2, 1, 2, 3, 2, 1, 2]
            Or, pad function with signature `pad_fn(x, pad_left, pad_right)`.
            This sets `self.pad_mode='custom'` (the name of padding is used
            for some internal logic).
        max_pad_factor : int (default 2), optional
            Will pad by at most `2**max_pad_factor` relative to `nextpow2(shape)`.
            E.g. if input length is 150, then maximum padding with
            `max_pad_factor=2` is `256 * (2**2) = 1024`.
        average_global_phi : bool
            True if `T == nextpow2(shape)`, i.e. `T` is maximum possible
            and equivalent to global averaging, in which case lowpassing is
            replaced by simple arithmetic mean.

            In case of `average==False`, controls scattering logic for
            `phi_t` pairs in JTFS.
        average_global : bool
            True if `average_global_phi and average_fr`. Same as
            `average_global_phi` if `average_fr==True`.

            In case of `average==False`, controls scattering logic for
            `psi_t` pairs in JTFS.
        r_psi : float / tuple[float]
            See `help(kymatio.scattering1d.utils.calibrate_scattering_filters)`.
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
        pad_mode : str (default 'reflect'), optional
            One of supported padding modes:
                - zero:    [0,0,0,0, 1,2,3,4, 0,0,0]
                - reflect: [3,4,3,2, 1,2,3,4, 3,2,1]
        max_pad_factor : int (default 2) / None, optional
            Will pad by at most `2**max_pad_factor` relative to `nextpow2(shape)`.
            E.g. if input length is 150, then maximum padding with
            `max_pad_factor=2` is `256 * (2**2) = 1024`.
            If None, won't restrict padding.
        analytic : bool (default False)
            If True, will force negative frequencies to zero. Useful if
            strict analyticity is desired, but may worsen time-domain decay.
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
        {param_shape}Q : int >= 1 / tuple[int]
            The number of first-order wavelets per octave. Defaults to `1`.
            If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
            second-order wavelets per octave (which defaults to `1`).
                - Q1: For audio signals, a value of `>= 12` is recommended in
                  order to separate partials.
                - Q2: Recommended `1` for most (`Scattering1D`) applications.
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
            may be modified after object creation.
            Defaults to `0`. Has no effect if `average_global=True`.
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


class TimeFrequencyScatteringBase1D():
    def __init__(self, J_fr=None, Q_fr=2, F=None, implementation=None,
                 average_fr=False, aligned=True,
                 sampling_filters_fr=('exclude', 'resample'),
                 max_pad_factor_fr=None, pad_mode_fr='conj-reflect-zero',
                 r_psi=math.sqrt(.5), oversampling_fr=0,
                 out_3D=False, out_type='array', out_exclude=None):
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.implementation = implementation
        self.average_fr = average_fr
        self.oversampling_fr = oversampling_fr
        self.aligned = aligned
        self.sampling_filters_fr = sampling_filters_fr
        self.max_pad_factor_fr = max_pad_factor_fr
        self.pad_mode_fr = pad_mode_fr
        self.r_psi_fr = r_psi
        self.out_3D = out_3D
        self.out_type = out_type
        self.out_exclude = out_exclude

    def build(self):
        """Check args and instantiate `_FrequencyScatteringBase` object
        (which builds filters).

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        # if config yields no second order coeffs, we cannot do joint scattering
        if self._no_second_order_filters:
            raise ValueError("configuration yields no second-order filters; "
                             "try increasing `J`")

        # handle `implementation`
        if (self.implementation is not None and
                self.implementation not in range(1, 6)):
            raise ValueError("`implementation` must be None or an integer 1-5 "
                             "(got %s)" % str(self.implementation))

        self._implementation_args = {
            1: dict(average_fr=False, aligned=True,  out_3D=False,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='array'),
            2: dict(average_fr=True,  aligned=True,  out_3D=True,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='array'),
            3: dict(average_fr=True,  aligned=True,  out_3D=True,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='dict:list'),
            4: dict(average_fr=True,  aligned=False, out_3D=True,
                    sampling_filters_fr=('exclude', 'recalibrate'),
                    out_type='array'),
            5: dict(average_fr=True,  aligned=False, out_3D=True,
                    sampling_filters_fr=('recalibrate', 'recalibrate'),
                    out_type='dict:list'),
        }
        defaults = self._implementation_args[1]
        for name in defaults:
            user_value = getattr(self, name)
            if self.implementation is None:
                if user_value is None:
                    setattr(self, name, defaults[name])
            else:
                if (name == 'out_type' and self.implementation in (3, 5)
                        and 'dict' in user_value):
                    implem_value = user_value
                else:
                    implem_value = self._implementation_args[self.implementation
                                                             ][name]
                setattr(self, name, implem_value)
        self.out_structure = (3 if self.implementation in (3, 5) else
                              None)

        # handle `out_exclude`
        if self.out_exclude is not None:
            if isinstance(self.out_exclude, str):
                self.out_exclude = [self.out_exclude]
            # ensure all names are valid
            supported = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                         'psi_t * phi_f', 'psi_t * psi_f_up',
                         'psi_t * psi_f_down')
            for name in self.out_exclude:
                if name not in supported:
                    raise ValueError(("'{}' is an invalid coefficient name; "
                                      "must be one of: {}").format(
                                          name, ', '.join(supported)))

        # handle `F`
        if self.F is None:
            # default to one octave (Q wavelets per octave, J octaves,
            # approx Q*J total frequency rows, so averaging scale is `Q/total`)
            # F is processed further in `_FrequencyScatteringBase`
            self.F = self.Q[0]

        # frequential scattering object ######################################
        self._shape_fr = self.get_shape_fr()
        # number of psi1 filters
        self._n_psi1_f = len(self.psi1_f)
        max_order_fr = 1

        self.sc_freq = _FrequencyScatteringBase(
            self._shape_fr, self.J_fr, self.Q_fr, self.F, max_order_fr,
            self.average_fr, self.aligned, self.oversampling_fr,
            self.sampling_filters_fr, self.out_type, self.out_3D,
            self.max_pad_factor_fr, self.pad_mode_fr, self.analytic,
            self.r_psi_fr, self._n_psi1_f, self.backend)
        self.finish_creating_filters()

        # detach __init__ args, instead access `sc_freq`'s via `__getattr__`
        # this is so that changes in attributes are reflected here
        init_args = ('J_fr', 'Q_fr', 'F', 'average_fr', 'oversampling_fr',
                     'sampling_filters_fr', 'max_pad_factor_fr', 'pad_mode_fr',
                     'r_psi_fr', 'out_3D')
        for init_arg in init_args:
            delattr(self, init_arg)

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
        # ensure phi is subsampled up to log2_T for `phi_t * psi_f` pairs
        max_sub_phi = lambda: max(k for k in self.phi_f if isinstance(k, int))
        while max_sub_phi() < self.log2_T:
            self.phi_f[max_sub_phi() + 1] = periodize_filter_fourier(
                self.phi_f[0], nperiods=2**(max_sub_phi() + 1))

        # for early unpadding in joint scattering
        # copy filters, assign to `0` trim (time's `subsample_equiv_due_to_pad`)
        phi_f = {0: [v for k, v in self.phi_f.items() if isinstance(k, int)]}
        # copy meta
        for k, v in self.phi_f.items():
            if not isinstance(k, int):
                phi_f[k] = v

        diff = min(self.J - self.log2_T, self.J_pad - self.N_scale)
        if diff > 0:
            for trim_tm in range(1, diff + 1):
                # subsample in Fourier <-> trim in time
                phi_f[trim_tm] = [v[::2**trim_tm] for v in phi_f[0]]
        self.phi_f = phi_f

        # adjust padding
        ind_start = {0: {k: v for k, v in self.ind_start.items()}}
        ind_end   = {0: {k: v for k, v in self.ind_end.items()}}
        if diff > 0:
            for trim_tm in range(1, diff + 1):
                pad_left, pad_right = compute_padding(self.J_pad - trim_tm,
                                                      self.N)
                start, end = compute_border_indices(
                    self.log2_T, self.J, pad_left, pad_left + self.N)
                ind_start[trim_tm] = start
                ind_end[trim_tm] = end
        self.ind_start, self.ind_end = ind_start, ind_end

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_jtfs()` with the parameters of the
        transform object.

        Returns
        ------
        meta : dictionary
            See `help(kymatio.scattering1d.utils.compute_meta_jtfs)`.
        """
        return compute_meta_jtfs(self.J_pad, self.J, self.Q, self.J_fr, self.Q_fr,
                                 self.T, self.F, self.aligned, self.out_3D,
                                 self.out_type, self.out_exclude,
                                 self.sampling_filters_fr, self.average,
                                 self.average_global, self.average_global_phi,
                                 self.oversampling, self.r_psi, self.sc_freq)

    @property
    def fr_attributes(self):
        """Exposes `sc_freq`'s attributes via main object."""
        return ('J_fr', 'Q_fr', 'shape_fr', 'shape_fr_max', 'shape_fr_min',
                'shape_fr_scale_max', 'shape_fr_scale_min',
                'J_pad_fr', 'J_pad_fr_max', 'J_pad_fr_max_init', 'average_fr',
                'average_fr_global', 'aligned', 'oversampling_fr', 'F', 'log2_F',
                'max_order_fr', 'max_pad_factor_fr', 'out_3D',
                'sampling_filters_fr', 'sampling_psi_fr', 'sampling_phi_fr',
                'phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_down')

    def __getattr__(self, name):
        # access key attributes via frequential class
        # only called when default attribute lookup fails
        # `hasattr` in case called from Scattering1D
        if name in self.fr_attributes and hasattr(self, 'sc_freq'):
            return getattr(self.sc_freq, name)
        raise AttributeError(f"'{type(self).__name__}' object has no "
                             f"attribute '{name}'")  # standard attribute error

    # docs ###################################################################
    @classmethod
    def _document(cls):
        cls.__doc__ = TimeFrequencyScatteringBase1D._doc_class.format(
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_call=cls._doc_alias_call,
            parameters=cls._doc_params,
            attributes=cls._doc_attrs,
            sample=cls._doc_sample.format(shape=cls._doc_shape),
            terminology=cls._terminology,
        )
        cls.scattering.__doc__ = (
            TimeFrequencyScatteringBase1D._doc_scattering.format(
                array=cls._doc_array,
                n=cls._doc_array_n,
        ))
        # doc `sc_freq` methods
        cls.sc_freq_compute_padding_fr.__doc__ = cls._doc_compute_padding_fr
        cls.sc_freq_compute_J_pad.__doc__ = cls._doc_compute_J_pad

    def output_size(self):
        raise NotImplementedError("Not implemented for JTFS.")

    def create_filters(self):
        raise NotImplementedError("Implemented in `_FrequencyScatteringBase`.")

    _doc_class = \
    r"""
    The 1D Joint Time-Frequency Scattering transform.

    JTFS builds on time scattering by convolving first order coefficients
    with joint 2D wavelets along time and frequency, increasing discriminability
    while preserving time-shift invariance. Invariance to frequency transposition
    can be imposed via frequential averaging, while preserving sensitivity to
    frequency-dependent time shifts.

    Joint wavelets are defined separably in time and frequency and permit fast
    separable convolution. Convolutions are followed by complex modulus and
    optionally averaging.

    The JTFS of a 1D signal :math:`x(t)` may be written as

        $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

    where

        $S_J^{{(0)}} x(t) = x \star \phi_T(t),$

        $S_J^{{(1)}} x(t, \lambda) =
        |x \star \psi_\lambda^{{(1)}}| \star \phi_T,$ and

        $S_J^{{(2)}} x(t, \lambda, \mu, l, s) =
        ||x \star \psi_\lambda^{{(1)}}| \star \Psi_{{\mu, l, s}}| \star \Phi_J.$

    $\Psi_{{\mu, l, s}}$ comprises of five kinds of joint wavelets:

        $\Psi_{{\mu, l, +1}}(t, \lambda) =
        \psi_\mu^{{(2)}}(t) \psi_{{l, s}}(-\lambda)$
        spin up bandpass

        $\Psi_{{\mu, l, -1}}(t, \lambda) =
        \psi_\mu^{{(2)}}(t) \psi_{{l, s}}(+\lambda)$
        spin down bandpass

        $\Psi_{{\mu, -\infty, 0}}(t, \lambda) =
        \psi_\mu^{{(2)}}(t) \phi_F(\lambda)$
        temporal bandpass, frequential lowpass

        $\Psi_{{-\infty, l, 0}}(t, \lambda) =
        \phi_T(t) \psi_{{l, s}}(\lambda)$
        temporal lowpass, frequential bandpass

        $\Psi_{{-\infty, -\infty, 0}}(t, \lambda)
        = \phi_T(t) \phi_F(\lambda)$
        joint lowpass

    and $\Phi_J$ optionally does temporal and/or frequential averaging:

        $\Phi_J(t, \lambda) = \phi_T(t) \phi_F(\lambda)$

    Above, :math:`\star` denotes convolution in time and/or frequency. The
    filters $\psi_\lambda^{{(1)}}(t)$ and $\psi_\mu^{{(2)}}(t)$ are analytic
    wavelets with center frequencies $\lambda$ and $\mu$, while
    $\phi_T(t)$ is a real lowpass filter centered at the zero frequency.
    $\psi_{{l, s}}(-\lambda)$ is like $\psi_\lambda^{{(1)}}(t)$ but with
    its own parameters (center frequency, support, etc), and an anti-analytic
    complement (spin down is analytic).

    Filters are built at initialization. While the wavelets are fixed, other
    parameters may be changed after the object is created, such as `out_type`.

    {frontend_paragraph}
    Example
    -------
    ::

        # Set the parameters of the scattering transform.
        J = 6
        N = 2 ** 13
        Q = 8

        # Generate a sample signal.
        x = {sample}

        # Define a `TimeFrequencyScattering1D` object.
        jtfs = TimeFrequencyScattering1D(J, N, Q)

        # Calculate the scattering transform.
        Sx = S.scattering(x)

        # Equivalently, use the alias.
        Sx = S{alias_call}(x)

    Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while the
    maximum scale of the scattering transform is set to :math:`2^J = 2^6 =
    64`. The time-frequency resolution of the first-order wavelets
    :math:`\psi_\lambda^{{(1)}}(t)` is set to `Q = 8` wavelets per octave.
    The second-order wavelets :math:`\psi_\mu^{{(2)}}(t)` have one wavelet
    per octave by default, but can be set like like `Q = (8, 2)`. Internally,
    `J_fr` and `Q_fr`, the frequential variants of `J` and `Q`, are defaulted,
    but can be specified as well.

    {parameters}

    {attributes}

    {terminology}
    """

    _doc_params = \
    r"""
    Parameters
    ----------
    J, shape, T, average, oversampling, pad_mode :
        See `help(kymatio.scattering1d.Scattering1D)`.

    Q : int / tuple[int]
        `(Q1, Q2)`, where `Q2=1` if `Q` is int. `Q1` is the number of first-order
        wavelets per octave, and `Q2` the second-order.

          - `Q1`, together with `J`, determines `shape_fr_max` and `shape_fr`,
            or length of inputs to frequential scattering.
          - `Q2`, together with `J`, determines `shape_fr` (via `j2 > j1`
            criterion), and total number of joint slices.
          - Greater `Q2` values better capture temporal AM modulations of
            multiple rates. Suited for inputs of multirate or intricate AM.
            `Q2=2` is in close correspondence with the mamallian auditory cortex:
            https://asa.scitation.org/doi/full/10.1121/1.1945807
            2 or 1 should work for most purposes.

    J_fr : int
        The maximum log-scale of frequential scattering in joint scattering
        transform, and number of octaves of frequential filters. That is,
        the maximum (bandpass) scale is given by `2**J_fr`.
        Default is determined at instantiation from longest frequential row
        in frequential scattering, set to `log2(nextpow2(shape_fr_max))`, i.e.
        maximum possible.

    Q_fr : int
        Number of wavelets per octave for frequential scattering.

        Greater values better capture quefrential variations of multiple rates
        - that is, variations and structures along frequency axis of the wavelet
        transform's 2D time-frequency plane. Suited for inputs of many frequencies
        or intricate AM-FM variations. 2 or 1 should work for most purposes.

    F : int / str['global'] / None
        Temporal support of frequential low-pass filter, controlling amount of
        imposed frequency transposition invariance and maximum frequential
        subsampling. Defaults to `Q`, i.e. one octave.

          - If `'global'`, sets to maximum possible `F` based on `shape_fr_max`.
          - Used even with `average_fr=False` (see its docs); this is likewise
            true of `T` for `phi_t * phi_f` and `phi_t * psi_f` pairs.

    implementation : int / None
        Preset configuration to use. Overrides the following parameters:

            - `average_fr, aligned, out_3D, sampling_filters_fr`

        Defaults to `None`, and any `None` argument above will default to
        that of `implementation=1`.
        See `help(kymatio.toolkit.pack_coeffs_jtfs)` for further information.

        **Implementations:**

            1: Standard for 1D convs. `(n1_fr * n2 * n1, t)`.
              - average_fr = False
              - aligned = True
              - out_3D = False
              - sampling_psi_fr = 'exclude'
              - sampling_phi_fr = 'resample'

            2: Standard for 2D convs. `(n1_fr * n2, n1, t)`.
              - average_fr = True
              - aligned = True
              - out_3D = True
              - sampling_psi_fr = 'exclude'
              - sampling_phi_fr = 'resample'

            3: Standard for 3D/4D convs. `(n1_fr, n2, n1, t)`. [2] but
              - out_structure = 3

            4: Efficient for 2D convs. [2] but
              - aligned = False
              - sampling_phi_fr = 'recalibrate'

            5: Efficient for 3D convs. [3] but
              - aligned = False
              - sampling_psi_fr = 'recalibrate'
              - sampling_phi_fr = 'recalibrate'

        `'exclude'` in `sampling_psi_fr` can be replaced with `'resample'`,
        which yields significantly more coefficients and is slower, but
        the coefficients are mostly "synthetic zeros" and uninformative.

        `out_structure` refers to packing output coefficients via
        `pack_coeffs_jtfs(..., out_structure)`. This zero-pads and reshapes
        coefficients, but does not affect their values or computation in any way.
        (Thus, 3==2 except for shape). Requires `out_type` 'dict:list' (default)
        or 'dict:array'; if 'dict:array' is passed, will use it instead.

    average_fr : bool (default False)
        Whether to average (lowpass) along frequency axis.

        If `False`, `phi_t * phi_f` and `psi_t * phi_f` pairs are still computed,
        but `psi_t * phi_f` aren't frequentially averaged the *second* time.

    aligned : bool (default True)
        If True, rows of joint slices index to same frequency for all slices.
        E.g. `S_2[3][5]` and `S_2[4][5]` (fifth row of third and fourth joint
        slices) correspond to same frequency. With `aligned=True`:

          - `out_3D=True`: all slices are zero-padded to have same number of rows.
            Earliest slices are likely to be mostly zero per `psi2` convolving
            with minority of first-order coefficients.
          - `out_3D=False`: all slices are padded by minimal amount needed to
            avert boundary effects.
              - `average_fr=True`: number of frequency rows will vary across
                slices but be same *per `psi2_f`*.
              - `average_fr=False`: number of rows will vary across and within
                slices (`psi1_f_fr_up`-to-`psi1_f_fr_up`, and down).

        If `aligned=False` with `out_3D=True`, will subsample all slices to same
        minimum controlled by `log2_F` (i.e. maximum subsampling); this breaks
        alignment but eliminates coeff redundancy
        (True oversamples relative to False).

        `aligned=True` works by forcing subsampling factors to be same in
        frequential scattering and frequential lowpassing across all joint slices;
        the factors are set to be same as in minimally padded case. This is
        because different `shape_fr` correspond to *trimming* of same underlying
        input (first order coeffs) rather than *subsampling*.

        Note: `sampling_psi_fr = 'recalibrate'` breaks global alignment, but
        preserves it on per-`subsample_equiv_due_to_pad` basis, i.e. per-`n2`.

        **Illustration**:

        `x` == zero, `0, 4, ...` == indices of actual (nonpadded) data
        ::

            data -> padded
            16   -> 128
            64   -> 128

            False:
              [0,  4,  8, 16,  x]
              [0, 16, 32, 48, 64]

            True:
              [0,  4,  8, 16,  x,  x,  x,  x,  x,  x,  x,  x,  x,  x,  x,  x]
              [0,  4,  8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

        `False` is more information dense, containing same information with
        fewer datapoints.

        In terms of unpadding with `out_3D=True`:
            - `aligned=True`: unpad subsampling factor decided from min
              padded case (as is `total_conv_stride_over_U1`); then maximum
              of all unpads across `n2` is taken for this factor and reused
              across `n2`.
            - `aligned=False`: decided from max padded case with max subsampling,
              and reused across `n2` (even with less subsampling).

    sampling_filters_fr : str / tuple[str]
        Controls filter properties for input lengths below maximum.

          - 'resample': preserve physical dimensionality (center frequeny, width)
            at every length (trimming in time domain).
            E.g. `psi = psi_fn(N/2) == psi_fn(N)[N/4:-N/4]`.
          - 'recalibrate': recalibrate filters to each length.
            - widths (in time): widest filter is halved in width, narrowest is
              kept unchanged, and other widths are re-distributed from the
              new minimum to same maximum.
            - center frequencies: all redistribute between new min and max.
              New min is set as `2 / new_length` (old min was `2 / max_length`).
              New max is set by halving distance between old max and 0.5
              (greatest possible), e.g. 0.44 -> 0.47, then 0.47 -> 0.485, etc.
          - 'exclude': same as 'resample' except filters wider than `widest / 2`
            are excluded. (and `widest / 4` for next `shape_fr_scale`, etc).

        Tuple can set separately `(sampling_psi_fr, sampling_phi_fr)`, else
        both set to same value.
        Also see `J_pad_fr_max` if `not all(sampling_filters_fr) == 'resample'`.

        From an information/classification standpoint:

            - 'resample' enforces freq invariance imposed by `phi_f_fr` and
              physical scale of extracted modulations by `psi1_f_fr_up` (& down).
              This is consistent with scattering theory and is the standard used
              in classification.
            - 'recalibrate' remedies a problem with 'resample'. 'resample'
              calibrates all filters relative to longest input; when the shortest
              input is very different in comparison, it makes most filters appear
              lowpass-ish. In contrast, recalibration enables better exploitation
              of fine structure over the smaller interval (which is the main
              motivation behind wavelets, a "multi-scale zoom".)
            - 'exclude' circumvents the problem by simply excluding wide filters.
              'exclude' is simply a subset of 'resample', preserving all center
              frequencies and widths - but 3D/4D concat is no longer possible.
              Preferred if not caring for 3D/4D concat.

        Note: `sampling_phi_fr = 'exclude'` will re-set to `'resample'`, as
        `'exclude'` isn't a valid option (there must exist a lowpass for every
        input length).

      max_pad_factor_fr : int / None (default) / list[int], optional
        `max_pad_factor` for frequential axis in frequential scattering.

            - None: unrestricted; will pad as much as needed.
            - list[int]: controls max padding for each `shape_fr_scale`
              separately, in reverse order (max to min).
                - Values must non-increasing (e.g. not `[2, 0, 1, ...]`)
                - If the list is insufficiently long (less than number of scales),
                  will extend list with the last provided value
                  (e.g. `[1, 2] -> [1, 2, 2, 2]`).
                - Indexed by `scale_diff == shape_fr_scale_max - shape_fr_scale`
            - int: will convert to list[int] of same value.

        Specified values aren't guaranteed to be realized. They override some
        padding values, but are overridden by others.

        Overrides:
            - Padding that lessens boundary effects and wavelet distortion
              (`min_to_pad`).

        Overridden by:
            - `J_pad_fr_min_limit_due_to_phi`
            - `J_pad_fr_min_limit_due_to_psi`
            - Will not allow any `J_pad_fr > J_pad_fr_max_init`
            - With `sampling_psi_fr = 'resample'`, will not allow `J_pad_fr`
              that yields a pure sinusoid wavelet (raises `ValueError` in
              `filter_bank.get_normalizing_factor`).

    pad_mode_fr : str (default 'conj-reflect-zero') / function
        Name of frequential padding mode to use, one of: 'zero',
        'conj-reflect-zero'.
        Or, function with signature `pad_fn_fr(x, pad_fr, sc_freq, B)`;
        see `_right_pad` in
        `kymatio.scattering1d.core.timefrequency_scattering1d`.

        If using `pad_mode = 'reflect'` and `average = True`, reflected portions
        will be automatically conjugated before frequential scattering to avoid
        spin cancellation. For same reason there isn't `pad_mode_fr = 'reflect'`.

    analytic : bool (default True)
        If True, will enforce strict analyticity/anti-analyticity:
            - zero negative frequencies for temporal and spin down bandpasses
            - zero positive frequencies for spin up bandpasses

        `True` is likely to improve FDTS-discriminability, especially for
        `r_psi > sqrt(.5)`, but may slightly worsen wavelet time decay.

    r_psi : float / tuple[float]
        See `help(kymatio.scattering1d.utils.calibrate_scattering_filters)`.
        Triple tuple sets `(r_psi1, r_psi2, r_psi_fr)`. If less than three
        are provided (included single float), last value is duplicated
        for the rest.

    oversampling_fr : int (default 0), optional
        How much to oversample along frequency axis (with respect to `2**J_fr`).
        Also see `oversampling` in `Scattering1D`.
        Has no effect if `average_fr_global=True`.

    out_3D : bool (default False)
        `True` (requires `average_fr=True`) adjusts frequential scattering
        to enable concatenation along joint slices dimension, as opposed to
        flattening (mixing slices and frequencies):

            - `False` will unpad freq by exact amounts for each joint slice,
              whereas `True` will unpad by minimum amount common to all
              slices at a given subsampling factor to enable concatenation.
              See `sc_freq_compute_padding_fr()`.
            - See `aligned` for its interactions with `out_3D`

        Both `True` and `False` can still be concatenated into the 'true' JTFS
        4D structure; see `help(kymatio.toolkit.pack_coeffs_jtfs)` for a complete
        description on coefficient structuring.

        The difference is in how values are computed, esp. near boundaries.
        From an information/classification standpoint,

          - `True` is more information-rich. The 1D equivalent case is unpadding
            by 3, instead of by 6 and then zero-padding by 3: same final length,
            but former fills gaps with partial convolutions where latter fills
            with zeros.
          - `False` is the "latter" case.

    out_type : str, optional
        Affects output structure (but not how coefficients are computed).
        See `help(TimeFrequencyScattering1D.scattering)` for further info.

            - 'list': coeffs are packed in a list of dictionaries, each dict
              storing meta info, and output tensor keyed by `'coef.`.
            - 'array': concatenated along slices (`out_3D=True`) or mixed
              slice-frequency dimension (`out_3D=False`). Both require
              `average=True` (and `out_3D=True` additionally `average_fr=True`).
            - 'dict:list' || 'dict:array': same as 'array' and 'list', except
              coefficients will not be concatenated across pairs - e.g. tensors
              from `'S1'` will be kept separate from those from `'phi_t * psi_f'`.
            - See `out_3D` for all behavior controlled by `out_3D`, and `aligned`
              for its behavior and interactions with `out_3D`.

    out_exclude : list/tuple[str] / None
        Will exclude coefficients with these names from computation and output
        (except for `S1`, which always computes but still excludes from output).
        All names:

            - 'S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
              'psi_t * psi_f_up', 'psi_t * psi_f_down'
    """

    _doc_attrs = \
    r"""
    Attributes
    ----------
    sc_freq : `_FrequencyScatteringBase`
        Frequential scattering object, storing pertinent attributes and filters.
        Temporal scattering's are accessed directly via `self`.

    shape_fr : list[int]
        List of lengths of frequential columns (i.e. numbers of frequential rows)
        in joint scattering, indexed by `n2` (second-order temporal wavelet idx).
        E.g. `shape_fr[3]==52` means 52 highest-frequency vectors from first-order
        time scattering are fed to `psi2_f[3]` (effectively, a multi-input
        network).

    shape_fr_max : int
        Equal to `max(shape_fr)`, used to set `J_pad_fr_max`.

    shape_fr_min : int
        `== min(shape_fr_realized)`

    shape_fr_realized: list[int]
        `shape_fr` without `0`s.

    shape_fr_max_all : int
        `== _n_psi1_f`. Used to compute `_J_pad_fr_fo` (unused quantity).

    shape_fr_scale : list[int]
        `== nextpow2(shape_fr)`. Fitlers are calibrated relative to these
        (for 'exclude' & 'recalibrate' `sampling_psi_fr`).

    shape_fr_scale_max : int
        `== max(shape_fr_scale)`.

            - Default value of `J_fr`. If `F == 2**J_fr`, then
              `average_fr_global=True`.
            - Used in `compute_J_pad()` and `psi_fr_factory()`.

    shape_fr_scale_min : int
        `== min(shape_fr_scale)`.

            - Used in `psi_fr_factory()` and `phi_fr_factory()`.

    phi_f_fr : dictionary
        Dictionary containing the frequential lowpass filter at all resolutions.
        See `help(kymatio.scattering1d.filter_bank.phi_fr_factory)`.

    psi1_f_fr_up : list[dict]
        List of dictionaries containing all frequential scattering filters
        with "up" spin.
        See `help(kymatio.scattering1d.filter_bank.psi_fr_factory)`.

    psi1_f_fr_down : list[dict]
        `psi1_f_fr_up`, but with "down" spin, forming a complementary pair.

    average_fr_global_phi : bool
        True if `F == nextpow2(shape_fr_max)`, i.e. `F` is maximum possible
        and equivalent to global averaging, in which case lowpassing is replaced
        by simple arithmetic mean.

        If True, `sampling_phi_fr` has no effect.

        In case of `average_fr==False`, controls scattering logic for
        `phi_f` pairs.

    average_fr_global : bool
        True if `average_fr_global_phi and average_fr`. Same as
        `average_fr_global_phi` if `average_fr==True`.

          - In case of `average_fr==False`, controls scattering logic for
            `psi_f` pairs.
          - If `True`, `phi_fr` filters are never used (but are still created).
          - Results are very close to lowpassing w/ `F == 2**shape_fr_scale_max`.
            Unlike with such lowpassing, `psi_fr` filters are allowed to be
            created at lower `J_pad_fr` than shortest `phi_fr` (which also is
            where greatest deviation with `not average_fr_global` occurs).

    log2_F : int
        Equal to `log2(prevpow2(F))`; is the maximum frequential subsampling
        factor if `average_fr=True` (otherwise that factor is up to `J_fr`).

    J_pad_fr : list[int]
        log2 of padding lengths of frequential columns in joint scattering
        (column lengths given by `shape_fr`). See `sc_freq.compute_padding_fr()`.

    J_pad_fr_max_init : int
        Set as reference for computing other `J_pad_fr`, is equal to
        `max(J_pad_fr)` with `unrestricted_pad_fr`.

    J_pad_fr_max : int
        `== max(J_pad_fr)`.

    J_pad_fr_min : int
        `== min(J_pad_fr)` (excluding -1).

    J_pad_fr_min_limit : int
        `J_pad_fr_min` cannot be less than this. Equals
        `max(J_pad_fr_min_limit_due_to_psi, J_pad_fr_min_limit_due_to_phi)`.

    J_pad_fr_min_limit_due_to_psi: int
        `== J_pad_fr_max_init - max_subsample_equiv_before_psi_fr`

    J_pad_fr_min_limit_due_to_phi : int
        `== J_pad_fr_max_init - max_subsample_equiv_before_phi_fr`

    min_to_pad_fr_max : int
        `min_to_pad` from `compute_minimum_support_to_pad(N=shape_fr_max)`.
        Used in computing `J_pad_fr`. See `sc_freq.compute_J_pad()`.

    unrestricted_pad_fr : bool
        `True` if `max_pad_factor is None`. Affects padding computation and
        filter creation:

          - `phi_f_fr` w/ `sampling_phi_fr=='resample'`:
            - `True`: will limit the shortest `phi_f_fr` to avoid distorting
              its time-domain shape
            - `False`: will compute `phi_f_fr` at every `J_pad_fr`
          - `psi_f_fr` w/ `sampling_psi_fr=='resample'`: same as phi

    subsample_equiv_relative_to_max_pad_init : int
        Amount of *equivalent subsampling* of frequential padding relative to
        `J_pad_fr_max_init`, indexed by `n2`.
        See `help(sc_freq.compute_padding_fr())`.

    max_subsample_equiv_before_phi_fr : int
        Maximum permitted `subsample_equiv_due_to_pad` before convolving
        with `phi_f_fr`, equal to `_n_phi_f_fr - 1` (unless `average_global_fr`,
        then unrestricted).

        This avoids distorting `phi`; temporal width is preserved with 'resample',
        so downsampling may yield a `phi` that never properly decays and wraps
        on itself.
          - Subsampling by `log2_F` *after* convolving with `phi_f_fr` is fine,
            thus the restriction is to not subsample by more than the most
            subsampled `phi_f_fr` *before* convolving with it.

    max_subsample_before_phi_fr : list[int]
        Maximum permitted `n1_fr_subsample`. May differ from `log2_F`
        if `sampling_phi_fr == 'recalibrate'`, in which case it accounts for
        (and varies with) `subsample_equiv_due_to_pad` (which the filter treats
        same as `n1_fr_subsample`; see "Returns" in
        `help(kymatio.scattering1d.filter_bank.phi_fr_factory)`).

        Difference with `max_subsample_equiv_before_phi_fr` is, this restricts
        `n1_fr_subsample` - the other restricts `subsample_equiv_due_to_pad`
        (i.e. sets `J_pad_fr_min_limit`).

    max_subsample_equiv_before_psi_fr : int / None
        Maximum permitted `subsample_equiv_due_to_pad` (equivalently, minimum
        permitted `J_pad_fr` as difference with `J_pad_fr_max_init`), i.e.
        shortest input psi can accept.

          - Is psi's equivalent of `max_subsample_equiv_before_phi_fr`
            (see its docs)
          - Is not `None` if `sampling_psi_fr in ('resample', 'recalibrate')`
          - With 'recalibrate', `max_subsample_equiv_before_psi_fr` will have
            no effect if build didn't terminate per `sigma_max_to_min_max_ratio`
            (i.e. it'll equal the largest existing `subsample_equiv_due_to_pad`).

    sigma_max_to_min_max_ratio : float >= 1
        Largest permitted `max(sigma) / min(sigma)`. Used with 'recalibrate'
        `sampling_psi_fr` to restrict how large the smallest sigma can get.

        Worst cases (high `subsample_equiv_due_to_pad`):
          - A value of `< 1` means a lower center frequency will have
            the narrowest temporal width, which is undesired.
          - A value of `1` means all center frequencies will have the same
            temporal width, which is undesired.
          - The `1.2` default was chosen arbitrarily as a seemingly good
            compromise between not overly restricting sigma and closeness to `1`.

    _n_phi_f_fr : int
        `== len(phi_f_fr)`. Used for setting `max_subsample_equiv_before_phi_fr`.

    pad_left_fr : int
        Amount of padding to left  of frequential columns
        (or top of joint matrix).

    pad_right_fr : int
        Amount of padding to right of frequential columns
        (or bottom of joint matrix).

    r_psi : tuple[float]
        Temporal redundancy, first- and second-order.

    r_psi_fr : float
        Frequential redundancy.
    """

    _terminology = \
    r"""
    Terminoloy
    ----------
    FDTS :
        Frequency-Dependent Time Shift. JTFS's main purpose is to detect these.
        Up spin wavelet resonates with up chirp (rising; right-shifts with
        increasing freq), down spin with down chirp (left-shifts with increasing
        freq).

        In convolution (cross-correlation with flipped kernel), the roles are
        reversed; the implementation will yield high values for up chirp
        from down spin.

    n1_fr_subsample, subsample_equiv_due_to_pad, n2 : int, int, int
        See `help(kymatio.scattering1d.core.timefrequency_scattering)`.
        Not attributes. Summary:

            - n1_fr_subsample: subsampling done after convolving with `psi_fr`
            - subsample_equiv_due_to_pad: "equivalent subsampling" due to padding
              less, equal to `J_pad_fr_max_init - pad_fr`
            - n2: index of temporal wavelet in joint scattering, like `psi2[n2]`.
    """

    _doc_scattering = \
    """
    Apply the Joint Time-Frequency Scattering transform.

    Given an input `{array}` of size `(B, N)`, where `B` is the batch size
    and `N` is the length of the individual signals, computes its JTFS.

    Output format is specified by `out_type`: a list, array, or dictionary of
    either with keys specifying coefficient names as follows:

    ::

        {{'S0': ...,                # (time)  zeroth order
         'S1': ...,                # (time)  first order
         'phi_t * phi_f': ...,     # (joint) joint lowpass
         'phi_t * psi_f': ...,     # (joint) time lowpass (w/ freq bandpass)
         'psi_t * phi_f': ...,     # (joint) freq lowpass (w/ time bandpass)
         'psi_t * psi_f_up': ...,  # (joint) spin up
         'psi_t * psi_f_up': ...,  # (joint) spin down
         }}

    Coefficient structure depends on `average, average_fr, aligned, out_3D`, and
    `sampling_filters_fr`. See `help(kymatio.toolkit.pack_coeffs_jtfs)` for a
    complete description.

    Parameters
    ----------
    x : {array}
        An input `{array}` of size `(B, N)` or `(N,)`.

    Returns
    -------
    S : dict[tensor/list]
        See above.
    """

    _doc_compute_padding_fr = \
    """
    Builds padding logic of frequential scattering.

      - `pad_left_fr, ind_start_fr`: always zero since we right-pad
      - `pad_right_fr`: computed to avoid boundary effects *for each* `shape_fr`.
      - `ind_end_fr`: computed to undo `pad_right_fr`
      - `subsample_equiv_relative_to_max_pad_init`: indexed by `n2`, is the
         amount of *equivalent subsampling* of padding relative to
         `J_pad_fr_max_init`.
         E.g.:
             - if `J_pad_fr_max_init=128` and we pad to 64 at `n2 = 3`, then
               `subsample_equiv_relative_to_max_pad_init[3] == 1`.
      - `ind_end_fr_max`: maximum unpad index across all `n2` for a given
        subsampling factor. E.g.:

        ::

            n2 = (0, 1, 2)
            J_fr = 4 --> j_fr = (0, 1, 2, 3)
            ind_end_fr = [[32, 16, 8, 4],
                          [29, 14, 7, 3],
                          [33, 16, 8, 4]]
            ind_end_fr_max = [33, 16, 8, 4]

        Ensures same unpadded freq length for `out_3D=True` without losing
        information. Unused for `out_3D=False`.

    Logic demos
    -----------
    For 'exclude' & 'recalibrate':
        1. Compute `psi`s at `J_pad_fr_max_init`
        2. Compute and store their 'width' meta
        3. Use width meta to compute padding for each `shape_fr`, excluding
           widths that exceed `nextpow2(shape_fr)`.
        4. Compute `psi`s at every other `J_pad_fr`

    Suppose `max_pad_factor_fr=0`, `J_pad_fr_max_init == 8`, and
    `nextpow2(shape_fr_max) == 7`. Then, for:
        'resample':
            - `J_pad_fr_max = 7`, and some `psi` will be distorted.
            - `subsample_equiv_due_to_pad` will now have a minimum value,
              `== J_pad_fr_max_init - (nextpow2(shape_fr_max) +
                                       max_pad_factor_fr)`,
              which determines `J_pad_fr` and thus `psi`'s length, as opposed
              to `max_pad_fr==None` case where `psi` determined max allowed
              `subsample_equiv_due_to_pad`.
            - But `psi`'s length will not be any shorter than
              "original_max - max_pad_factor_fr", i.e.
              `J_pad_fr_max_init - max_pad_factor_fr`.
            - Thus `J_pad_fr_min == J_pad_fr_max_init - max_pad_factor_fr`.

        'exclude':
            - `J_pad_fr_max = 7`, and some `psi` will be distorted.
            - `J_pad_fr = min(J_pad_fr_original,
                              nextpow2(shape_fr) + max_pad_factor_fr)`
            - `psi`s will still be excluded based on `shape_fr` alone,
              independent of `J_pad_fr`, so some `psi` will be distorted.

        'recalibrate':
            - `J_pad_fr_max = 7`, and some `psi` will be distorted.
            - `J_pad_fr = min(J_pad_fr_original,
                              nextpow2(shape_fr) + max_pad_factor_fr)`
            - Lowest sigma `psi` will still be set based on `shape_fr` alone,
              independent of `J_pad_fr`, so some `psi` will be distorted.

    `psi` calibration and padding
    -----------------------------
    In computing padding, we take `nextpow2(shape_fr)` for every `shape_fr`.
    This guarantees no two `shape_fr` from different `shape_fr_scale` have
    the same `J_pad_fr`, else we'd need triple indexed wavelets (non-'resample'):
    `psi_fr[n1_fr][n2][subsample_equiv_due_to_pad]`. When `J_fr` is at
    or one below maximum, this happens automatically, so this only loses speed
    relative to small `J_fr`.

    Illustrating, consider `shape_fr, width(psi_fr_widest) -> J_pad_fr`:
    ::

        33, 16 -> 64   # J_fr == 2 below max
        33, 32 -> 128  # J_fr == 1 below max
        33, 64 -> 128  # J_fr == max
        63, 64 -> 128  # J_fr == max

    Non-uniqueness example, w/ # `(J_pad_fr, shape_fr_scale)`:
    ::

        33, 4 -> 64  # `(6, 6)`
        31, 2 -> 64  # `(5, 6)`
        17, 2 -> 32  # `(5, 5)`
    """

    _doc_compute_J_pad = \
    """
    Depends on `shape_fr`, `sampling_phi_fr`, and `sampling_psi_fr`.

    `min_to_pad` is computed for both `phi` and `psi` in case latter has greater
    time-domain support (stored as `_pad_fr_phi` and `_pad_fr_psi`).
      - 'resample': will use original `_pad_fr_phi` and/or `_pad_fr_psi`
      - 'recalibrate' / 'exclude': will divide by difference in dyadic scale,
        e.g. `_pad_fr_phi / 2`.

    `recompute=True` will force computation from `shape_fr` alone, independent
    of `J_pad_fr_max` and `min_to_pad_fr_max`, and per `resample_* == True`.
    """


class _FrequencyScatteringBase(ScatteringBase):
    """Attribute object for TimeFrequencyScatteringBase1D for frequential
    scattering part of JTFS.
    """
    def __init__(self, shape_fr, J_fr=None, Q_fr=2, F=None, max_order_fr=1,
                 average_fr=False, aligned=True, oversampling_fr=0,
                 sampling_filters_fr='resample', out_type='array', out_3D=False,
                 max_pad_factor_fr=None, pad_mode_fr='conj-reflect-zero',
                 analytic=True, r_psi_fr=math.sqrt(.5), n_psi1=None,
                 backend=None):
        super(_FrequencyScatteringBase, self).__init__()
        self.shape_fr = shape_fr
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.max_order_fr = max_order_fr
        self.average_fr = average_fr
        self.aligned = aligned
        self.oversampling_fr = oversampling_fr
        self.sampling_filters_fr = sampling_filters_fr
        self.sampling_psi_fr = None  # set in build()
        self.sampling_phi_fr = None  # set in build()
        self.out_type = out_type
        self.out_3D = out_3D
        self.max_pad_factor_fr = max_pad_factor_fr
        self.pad_mode_fr = pad_mode_fr
        self.analytic_fr = analytic
        self.r_psi_fr = r_psi_fr
        self._n_psi1_f = n_psi1
        self.backend = backend

        self.build()
        self.create_phi_filters()
        self.compute_padding_fr()
        self.create_psi_filters()
        self.adjust_padding_and_filters()

    def build(self):
        self.sigma0 = 0.1
        self.alpha = 4.
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3
        self.normalize = 'l1'
        self.sigma_max_to_min_max_ratio = 1.2

        # `shape_fr` used in scattering, == realized `psi2_f`s
        self.shape_fr_realized = [s for s in self.shape_fr if s > 0]
        # longest & shortest obtainable frequency row w.r.t. which we
        # calibrate filters
        self.shape_fr_max = max(self.shape_fr_realized)
        self.shape_fr_min = min(self.shape_fr_realized)
        # above is for `psi_t *` pairs, below is actual max, which
        # occurs for `phi_t *` pairs
        self.shape_fr_max_all = self._n_psi1_f
        # compute corresponding scales
        self.shape_fr_scale = [(math.ceil(math.log2(s)) if s != 0 else -1)
                               for s in self.shape_fr]
        self.shape_fr_scale_max = max(self.shape_fr_scale)
        # smallest scale is also smallest possible maximum padding
        # (cannot be overridden by `max_pad_factor_fr`)
        self.shape_fr_scale_min = min(s for s in self.shape_fr_scale if s != -1)
        # store number of unique scales
        self.n_scales_fr = len(np.unique([s for s in self.shape_fr_scale
                                          if s != -1]))

        # ensure 2**J_fr <= nextpow2(shape_fr_max)
        if self.J_fr is None:
            self.J_fr = self.shape_fr_scale_max
        elif self.J_fr > self.shape_fr_scale_max:
            raise ValueError(("2**J_fr cannot exceed maximum number of frequency "
                              "rows (rounded up to pow2) in joint scattering "
                              "(got {} > {})".format(
                                  2**(self.J_fr), 2**self.shape_fr_scale_max)))

        # check F or set default
        if self.F == 'global':
            self.F = 2**self.shape_fr_scale_max
        elif self.F > 2**self.shape_fr_scale_max:
            raise ValueError("The temporal support F of the low-pass filter "
                             "cannot exceed maximum number of frequency rows "
                             "(rounded up to pow2) in joint scattering "
                             "(got {} > {})".format(
                                 self.F, 2**self.shape_fr_scale_max))
        self.log2_F = math.floor(math.log2(self.F))
        self.average_fr_global_phi = bool(self.F == 2**self.shape_fr_scale_max)
        self.average_fr_global = bool(self.average_fr_global_phi and
                                      self.average_fr)

        # restrict `J_pad_fr_max` (and `J_pad_fr_max_init`) if specified by user
        err = ValueError("`max_pad_factor_fr` must be int>0, non-increasing "
                         "list/tuple[int>0], or None " +
                         "(got %s)" % str(self.max_pad_factor_fr))
        if isinstance(self.max_pad_factor_fr, int):
            self.max_pad_factor_fr = [self.max_pad_factor_fr
                                      ] * self.n_scales_fr

        elif isinstance(self.max_pad_factor_fr, (list, tuple)):
            if any(np.diff(self.max_pad_factor_fr) > 0):
                raise err
            while len(self.max_pad_factor_fr) < self.n_scales_fr:
                # repeat last value
                self.max_pad_factor_fr.append(self.max_pad_factor_fr[-1])
            # must guarantee that `J_pad_fr > J_pad_fr_max_init` cannot occur
            if max(self.max_pad_factor_fr) > self.max_pad_factor_fr[0]:
                J_pad_fr_max = max(s + p for s, p in
                                   zip(self.shape_fr_scale[::-1],
                                       self.max_pad_factor_fr))
                first_max_pf_min = J_pad_fr_max - self.shape_fr_scale_max
                self.max_pad_factor_fr[0] = first_max_pf_min

        elif self.max_pad_factor_fr is None:
            self.J_pad_fr_max_user = None
        else:
            raise err
        self.unrestricted_pad_fr = bool(self.max_pad_factor_fr is None)

        # check `pad_mode_fr`, set `pad_fn_fr`
        supported = ('conj-reflect-zero', 'zero')
        if isinstance(self.pad_mode_fr, FunctionType):
            fn = self.pad_mode_fr
            def pad_fn_fr(x, pad_fr, sc_freq, B):
                return fn(x, pad_fr, sc_freq, B)
            self.pad_mode_fr = 'custom'
        elif self.pad_mode_fr not in supported:
            raise ValueError(("unsupported `pad_mode_fr` '{}';\nmust be a "
                              "function, or string, one of: {}").format(
                                  self.pad_mode_fr, ', '.join(supported)))
        else:
            pad_fn_fr = None  # handled in `core`
        self.pad_fn_fr = pad_fn_fr

        # unpack `sampling_` args
        if isinstance(self.sampling_filters_fr, tuple):
            self.sampling_psi_fr, self.sampling_phi_fr = self.sampling_filters_fr
            if self.sampling_phi_fr == 'exclude':
                # if user explicitly passed 'exclude' for `_phi`
                warnings.warn("`sampling_phi_fr = 'exclude'` has no effect, "
                              "will use 'resample' instead.")
                self.sampling_phi_fr = 'resample'
        else:
            self.sampling_psi_fr = self.sampling_phi_fr = self.sampling_filters_fr
            if self.sampling_phi_fr == 'exclude':
                self.sampling_phi_fr = 'resample'
        self.sampling_filters_fr = (self.sampling_psi_fr, self.sampling_phi_fr)

        # validate `sampling_*` args
        psi_supported = ('resample', 'recalibrate', 'exclude')
        phi_supported = ('resample', 'recalibrate')
        if self.sampling_psi_fr not in psi_supported:
            raise ValueError(("unsupported `sampling_psi_fr` ({}), must be one "
                              "of: {}").format(self.sampling_psi_fr,
                                               ', '.join(psi_supported)))
        elif self.sampling_phi_fr not in phi_supported:
            raise ValueError(("unsupported `sampling_phi_fr` ({}), must be one "
                              "of: {}").format(self.sampling_phi_fr,
                                               ', '.join(phi_supported)))

        # compute maximum amount of padding
        # we do this at max possible `shape_fr` per each dyadic scale
        # to guarantee pad uniqueness across scales; see `compute_padding_fr` docs
        (self.J_pad_fr_max_init, self.min_to_pad_fr_max, self._pad_fr_phi,
         self._pad_fr_psi) = self._compute_J_pad(2**self.shape_fr_scale_max,
                                                 Q=(self.Q_fr, 0))

    def create_phi_filters(self):
        """See `filter_bank.phi_fr_factory`."""
        self.phi_f_fr = phi_fr_factory(
            self.J_pad_fr_max_init, self.F, self.log2_F,
            **self.get_params('shape_fr_scale_min', 'shape_fr_scale_max',
                              'unrestricted_pad_fr', 'sampling_phi_fr',
                              'criterion_amplitude', 'sigma0', 'P_max', 'eps'))

        # if we pad less, `phi_f_fr[subsample_equiv_due_to_pad]` fails
        # Overrides `max_pad_factor_fr`.
        self._n_phi_f_fr = len([k for k in self.phi_f_fr if isinstance(k, int)])
        self.max_subsample_equiv_before_phi_fr = (
            self._n_phi_f_fr - 1 if not self.average_fr_global else
            self.J_pad_fr_max_init)

    def create_psi_filters(self):
        """See `filter_bank.psi_fr_factory`."""
        (self.psi1_f_fr_up, self.psi1_f_fr_down,
         self.max_subsample_equiv_before_psi_fr) = psi_fr_factory(
            self.J_pad_fr_max_init, self.J_fr, self.Q_fr, self.shape_fr,
            **self.get_params(
                'shape_fr_scale_max', 'shape_fr_scale_min', 'pad_mode_fr',
                'max_pad_factor_fr', 'unrestricted_pad_fr',
                'max_subsample_equiv_before_phi_fr',
                'subsample_equiv_relative_to_max_pad_init',
                'average_fr_global_phi', 'sampling_psi_fr', 'sampling_phi_fr',
                'sigma_max_to_min_max_ratio',
                'r_psi_fr', 'normalize', 'sigma0', 'alpha', 'P_max', 'eps'))

        # cannot do energy norm with 3 filters, and generally filterbank
        # isn't well-behaved
        n_psi_frs = len(self.psi1_f_fr_up)
        if n_psi_frs <= 3:
            raise Exception(("configuration yielded %s wavelets for frequential "
                             "scattering, need a minimum of 4; try increasing "
                             "J, Q, J_fr, or Q_fr." % n_psi_frs))

        # analyticity
        if self.analytic_fr:
            psi_fs_all = (self.psi1_f_fr_up, self.psi1_f_fr_down)
            for s1_fr, psi_fs in enumerate(psi_fs_all):
              for n1_fr in range(len(psi_fs)):
                for j0 in psi_fs[n1_fr]:
                  if isinstance(j0, int):
                      pf = psi_fs[n1_fr][j0]
                      M = len(pf)
                      if s1_fr == 0:
                          pf[:M//2] = 0      # anti-analytic, zero positives
                      else:
                          pf[M//2 + 1:] = 0  # analytic, zero negatives
                      pf[M//2] /= 2          # halve Nyquist

    def adjust_padding_and_filters(self):
        # adjust padding
        if self.max_subsample_equiv_before_psi_fr is not None:
            self.J_pad_fr_min_limit_due_to_psi = (
                self.J_pad_fr_max_init - self.max_subsample_equiv_before_psi_fr)
            self.J_pad_fr_min_limit = max(self.J_pad_fr_min_limit_due_to_phi,
                                          self.J_pad_fr_min_limit_due_to_psi)
            # adjust existing J_pad_fr per (potentially) new J_pad_fr_min_limit;
            # overrides `max_pad_factor_fr`.
            for n2, (J_pad_fr, shape_fr
                     ) in enumerate(zip(self.J_pad_fr, self.shape_fr)):
                if J_pad_fr != -1:
                    if self.unrestricted_pad_fr:
                        J_pad_fr = max(J_pad_fr, self.J_pad_fr_min_limit)
                    else:
                        # `min` not needed in current implem as it's already
                        # realized in `compute_J_pad()` but keep to be sure
                        shape_fr_scale = math.ceil(math.log2(shape_fr))
                        scale_diff = self.shape_fr_scale_max - shape_fr_scale
                        J_pad_fr = max(min(J_pad_fr,
                                           shape_fr_scale +
                                           self.max_pad_factor_fr[scale_diff]),
                                       self.J_pad_fr_min_limit)

                    self.J_pad_fr[n2] = J_pad_fr
                    j0, pad_left, pad_right, ind_start, ind_end = (
                        self._compute_padding_params(J_pad_fr, shape_fr))
                    self.subsample_equiv_relative_to_max_pad_init[n2] = j0
                    self.pad_left_fr[n2] = pad_left
                    self.pad_right_fr[n2] = pad_right
                    self.ind_start_fr[n2] = ind_start
                    self.ind_end_fr[n2] = ind_end
        else:
            self.J_pad_fr_min_limit_due_to_psi = None
            self.J_pad_fr_min_limit = self.J_pad_fr_min_limit_due_to_phi
        # validate resulting J_pad_fr
        prev_pad = -2
        for pad in self.J_pad_fr:
            if pad < prev_pad:
                raise Exception("`max_pad_factor_fr` yielded padding that's "
                                "greater for lesser `shape_fr_scale`; this is "
                                "likely to yield incorrect or undefined behavior."
                                "\nJ_pad_fr=%s" % self.J_pad_fr)
            prev_pad = pad
        # realized minimum
        self.J_pad_fr_min = min(p for p in self.J_pad_fr if p != -1)

        if not self.unrestricted_pad_fr:
            # adjust phi_fr
            j0_max_realized = self.J_pad_fr_max_init - self.J_pad_fr_min
            j_frs = [k for k in self.phi_f_fr if isinstance(k, int)]
            for j_fr in j_frs:
                if j_fr > j0_max_realized:
                    del self.phi_f_fr[j_fr]

        # energy norm
        energy_norm_filterbank_fr(self.psi1_f_fr_up, self.psi1_f_fr_down,
                                  self.phi_f_fr, self.J_fr, self.log2_F)

    def compute_padding_fr(self):
        """Docs in `TimeFrequencyScatteringBase1D`."""
        # tentative limit
        self.J_pad_fr_min_limit_due_to_phi = (
            self.J_pad_fr_max_init - self.max_subsample_equiv_before_phi_fr)

        attrs = ('J_pad_fr', 'pad_left_fr', 'pad_right_fr',
                 'ind_start_fr', 'ind_end_fr',
                 'ind_start_fr_max', 'ind_end_fr_max',
                 'subsample_equiv_relative_to_max_pad_init')
        for attr in attrs:
            setattr(self, attr, [])

        # J_pad is ordered lower to greater, so iterate backward then reverse
        # (since we don't yet know max `j0`)
        for n2_reverse, shape_fr in enumerate(self.shape_fr[::-1]):
            if shape_fr != 0:
                J_pad = self.compute_J_pad(shape_fr)

                # compute the padding quantities
                j0, pad_left, pad_right, ind_start, ind_end = (
                    self._compute_padding_params(J_pad, shape_fr))
            else:
                J_pad, pad_left, pad_right, j0 = -1, -1, -1, -1
                ind_start, ind_end = [], []

            self.J_pad_fr.append(J_pad)
            self.pad_left_fr.append(pad_left)
            self.pad_right_fr.append(pad_right)
            self.ind_start_fr.append(ind_start)
            self.ind_end_fr.append(ind_end)
            self.subsample_equiv_relative_to_max_pad_init.append(j0)

        for attr in attrs:
            getattr(self, attr).reverse()
        self.J_pad_fr_max = max(self.J_pad_fr)

        # compute maximum ind_start and ind_end across all subsampling factors
        # to use as common for out_type='array'
        def get_idxs(attr):
            return getattr(self, attr.strip('_max'))

        for attr in ('ind_start_fr_max', 'ind_end_fr_max'):
            for j in range(self.log2_F + 1):
                idxs_max = max(get_idxs(attr)[n2][j]
                               for n2 in range(len(self.shape_fr))
                               if len(get_idxs(attr)[n2]) != 0)
                getattr(self, attr).append(idxs_max)

        # max `n1_fr_subsample`; this may vary with `subsample_equiv_due_to_pad`
        self.max_subsample_before_phi_fr = []
        for n2, shape_fr in enumerate(self.shape_fr):
            j0 = self.subsample_equiv_relative_to_max_pad_init[n2]
            if j0 == -1:
                sub = -1
            elif self.average_fr_global_phi:
                # "lowpass shape" always same, can't distort
                sub = 999
            else:
                sub = len(self.phi_f_fr[j0]) - 1
            self.max_subsample_before_phi_fr.append(sub)

        # unused quantity; if this exceeds `J_pad_fr_max_init`, then
        # `phi_t * psi_f` and `phi_t * phi_f` pairs will incur boundary effects.
        # Implem doesn't account for this as the effect is rare and most often
        # not great, while greatly complicating implem logic
        self._J_pad_fr_fo = self.compute_J_pad(self.shape_fr_max_all,
                                               recompute=True, Q=(0, 0))

    def _compute_padding_params(self, J_pad, shape_fr):
        pad_left = 0
        pad_right = 2**J_pad - pad_left - shape_fr
        j0 = self.J_pad_fr_max_init - J_pad
        assert j0 >= 0, "%s > %s | %s" % (J_pad, self.J_pad_fr_max_init, shape_fr)

        # compute unpad indices for all possible subsamplings
        ind_start, ind_end = [0], [shape_fr]
        for j in range(1, max(self.J_fr, self.log2_F) + 1):
            ind_start.append(0)
            ind_end.append(math.ceil(ind_end[-1] / 2))
        return j0, pad_left, pad_right, ind_start, ind_end

    def compute_J_pad(self, shape_fr, recompute=False, Q=(0, 0)):
        """Docs in `TimeFrequencyScatteringBase1D`."""
        # for later
        shape_fr_scale = int(np.ceil(np.log2(shape_fr)))
        scale_diff = self.shape_fr_scale_max - shape_fr_scale
        factor = 2**scale_diff
        shape_fr_max_at_scale = 2**shape_fr_scale

        if recompute:
            J_pad, *_ = self._compute_J_pad(shape_fr_max_at_scale, Q)
        elif (self.sampling_phi_fr == 'resample' or
              self.sampling_psi_fr == 'resample'):
            if self.sampling_phi_fr == 'resample':
                if self.sampling_psi_fr == 'resample':
                    min_to_pad = self.min_to_pad_fr_max
                else:
                    # see `else` below
                    min_to_pad = max(self._pad_fr_phi, self._pad_fr_psi // factor)
            else:
                # 'exclude' is equivalent to 'recalibrate' in terms of min sigma
                min_to_pad = max(self._pad_fr_psi, self._pad_fr_phi // factor)
            J_pad = math.ceil(np.log2(shape_fr_max_at_scale + 2 * min_to_pad))
        else:
            # both 'recalibrate' and 'exclude' build precisely to this logic
            min_to_pad = self.min_to_pad_fr_max // factor
            J_pad = math.ceil(np.log2(shape_fr_max_at_scale + 2 * min_to_pad))

        # don't let J_pad exceed user-set max
        if not self.unrestricted_pad_fr:
            J_pad = min(J_pad,
                        shape_fr_scale + self.max_pad_factor_fr[scale_diff])
        # don't let J_pad drop below minimum
        J_pad = max(J_pad, self.J_pad_fr_min_limit_due_to_phi)
        return J_pad

    def _compute_J_pad(self, shape_fr, Q):
        min_to_pad, pad_phi, pad_psi1, _ = compute_minimum_support_to_pad(
            shape_fr, self.J_fr, Q, self.F, pad_mode=self.pad_mode_fr,
            r_psi=self.r_psi_fr,
            **self.get_params( 'sigma0', 'alpha', 'P_max', 'eps',
                              'criterion_amplitude', 'normalize'))
        if self.average_fr_global_phi:
            min_to_pad = pad_psi1  # ignore phi's padding
            pad_phi = 0
        J_pad = math.ceil(np.log2(shape_fr + 2 * min_to_pad))

        if not self.unrestricted_pad_fr:
            shape_fr_scale = math.ceil(math.log2(shape_fr))
            scale_diff = self.shape_fr_scale_max - shape_fr_scale
            J_pad = min(J_pad,
                        shape_fr_scale + self.max_pad_factor_fr[scale_diff])
        return J_pad, min_to_pad, pad_phi, pad_psi1

    def get_params(self, *args):
        return {k: getattr(self, k) for k in args}


def _check_runtime_args_jtfs(average, average_fr, out_type, out_3D):
    if 'array' in out_type and not average:
        raise ValueError("Options `average=False` and `'array' in out_type` "
                         "are mutually incompatible. "
                         "Please set out_type='list' or 'dict:list'")

    if out_3D and not average_fr:
        raise ValueError("`out_3D=True` requires `average_fr=True`.")

    supported = ('array', 'list', 'dict:array', 'dict:list')
    if out_type not in supported:
        raise RuntimeError("`out_type` must be one of: {} (got {})".format(
            ', '.join(supported), out_type))

def _handle_args_jtfs(oversampling, oversampling_fr, r_psi, out_type):
    # handle defaults
    if oversampling_fr is None:
        oversampling_fr = oversampling
    if isinstance(r_psi, tuple):
        if len(r_psi) == 2:
            r_psi = (*r_psi, r_psi[-1])
    else:
        r_psi = (r_psi, r_psi, r_psi)
    r_psi_tm = r_psi[:2]
    r_psi_fr = r_psi[-1]

    # Second-order scattering object for the time variable
    max_order_tm = 2
    scattering_out_type = out_type.lstrip('dict:')
    return oversampling_fr, r_psi_tm, r_psi_fr, max_order_tm, scattering_out_type


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase1D']
