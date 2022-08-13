from ...frontend.base_frontend import ScatteringBase
import math
import numbers
import numpy as np
from warnings import warn

from ..filter_bank import compute_temporal_support, gauss_1d, scattering_filter_factory
from ..utils import (compute_border_indices, compute_padding,
compute_meta_scattering, precompute_size_scattering)


class ScatteringBase1D(ScatteringBase):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=None,
                 oversampling=0, out_type='array', backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.out_type = out_type
        self.backend = backend

        if average is not None:
            warn("The average option is deprecated and will be removed in v0.4."
                 " For average=True, set T=None for default averaging"
                 " or T>=1 for custom averaging."
                 " For average=False set T=0.",
                 DeprecationWarning)

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
            raise ValueError('Q should always be >= 1, got {}'.format(self.Q))

        if isinstance(self.Q, int):
            self.Q = (self.Q, 1)
        elif isinstance(self.Q, tuple):
            if len(self.Q) == 1:
                self.Q = self.Q + (1, )
            elif len(self.Q) < 1 or len(self.Q) > 2:
                raise NotImplementedError("Q should be an integer, 1-tuple or "
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
        if self.T is None:
            self.T = 2 ** self.J
            self.average = True if self.average is None else self.average
        elif self.T > N_input:
            raise ValueError("The temporal support T of the low-pass filter "
                             "cannot exceed input length (got {} > {})".format(
                                 self.T, N_input))
        elif self.T == 0:
            if not self.average:
                self.T = 2 ** self.J
                self.average = False
            else:
                raise ValueError("average must not be True if T=0 "
                                 "(got {})".format(self.average))
        elif self.T < 1:
            raise ValueError("T must be ==0 or >=1 (got {})".format(
                             self.T))
        else:
            self.average = True if self.average is None else self.average
            if not self.average:
                raise ValueError("average=False is not permitted when T>=1, "
                                 "(got {}). average is deprecated in v0.3 in "
                                 "favour of T and will "
                                 "be removed in v0.4.".format(self.T))


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
            self._N_padded, self.J, self.Q, self.T,
            r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha)
        ScatteringBase._check_filterbanks(self.psi1_f, self.psi2_f)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return compute_meta_scattering(
            self.J, self.Q, self.T, self.max_order, self.r_psi, self.sigma0, self.alpha)

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
        size = precompute_size_scattering(self.J, self.Q, self.T,
            self.max_order, self.r_psi, self.sigma0, self.alpha)
        if not detail:
            size = sum(size)
        return size

    def _check_runtime_args(self):
        if not self.out_type in ('array', 'dict', 'list'):
            raise ValueError("The out_type must be one of 'array', 'dict'"
                             ", or 'list'. Got: {}".format(self.out_type))

        if not self.average and self.out_type == 'array':
            raise ValueError("Cannot convert to out_type='array' with "
                             "average=False. Please set out_type to 'dict' or 'list'.")

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


__all__ = ['ScatteringBase1D']
