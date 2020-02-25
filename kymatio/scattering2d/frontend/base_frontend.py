from ...frontend.base_frontend import ScatteringBase

from ..filter_bank import filter_bank
from ..utils import compute_padding


class ScatteringBase2D(ScatteringBase):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend=None, out_type='array'):
        super(ScatteringBase2D, self).__init__()
        self.pre_pad = pre_pad
        self.L = L
        self.backend = backend
        self.J = J
        self.shape = shape
        self.max_order = max_order
        self.out_type = out_type

    def build(self):
        self.M, self.N = self.shape

        if 2 ** self.J > self.M or 2 ** self.J > self.N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = self.backend.Pad([(self.M_padded - self.M) // 2, (self.M_padded - self.M+1) // 2, (self.N_padded - self.N) // 2,
                                (self.N_padded - self.N + 1) // 2], [self.M, self.N], pre_pad=self.pre_pad)
        self.unpad = self.backend.unpad

    def create_filters(self):
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

    _doc_shape = 'M, N'

    _doc_instantiation_shape = {True: 'S = Scattering2D(J, (M, N))',
                                False: 'S = Scattering2D(J)'}

    _doc_param_shape = \
    r"""shape : tuple of ints
            Spatial support (M, N) of the input
        """

    _doc_attrs_shape = \
    r"""Psi : dictionary
            Contains the wavelets filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        Phi : dictionary
            Contains the low-pass filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        M_padded, N_padded : int
             Spatial support of the padded input.
        """

    _doc_param_out_type = \
    r"""out_type : str, optional
            The format of the output of a scattering transform. If set to
            `'list'`, then the output is a list containing each individual
            scattering path with meta information. Otherwise, if set to
            `'array'`, the output is a large array containing the
            concatenation of all scattering coefficients. Defaults to
            `'array'`.
        """

    _doc_attr_out_type = \
    r"""out_type : str
            The format of the scattering output. See documentation for
            `out_type` parameter above and the documentation for `scattering`.
        """

    _doc_class = \
    r"""The 2D scattering transform

        The scattering transform computes two wavelet transform
        followed by modulus non-linearity. It can be summarized as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x = x \star \phi_J$,

            $S_J^{{(1)}} x = [|x \star \psi^{{(1)}}_\lambda| \star \phi_J]_\lambda$, and

            $S_J^{{(2)}} x = [||x \star \psi^{{(1)}}_\lambda| \star
            \psi^{{(2)}}_\mu| \star \phi_J]_{{\lambda, \mu}}$.

        where $\star$ denotes the convolution (in space), $\phi_J$ is a
        lowpass filter, $\psi^{{(1)}}_\lambda$ is a family of bandpass filters
        and $\psi^{{(2)}}_\mu$ is another family of bandpass filters. Only
        Morlet filters are used in this implementation. Convolutions are
        efficiently performed in the Fourier domain.
        {frontend_paragraph}
        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N = 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering2D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Parameters
        ----------
        J : int
            Log-2 of the scattering scale.
        {param_shape}L : int, optional
            Number of angles used for the wavelet transform. Defaults to `8`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally. Defaults to `False`.
        backend : object, optional
            Controls the backend which is combined with the frontend.
        {param_out_type}
        Attributes
        ----------
        J : int
            Log-2 of the scattering scale.
        {param_shape}L : int, optional
            Number of angles used for the wavelet transform.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either `1` or `2`.
        pre_pad : boolean
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally.
        {attrs_shape}{attr_out_type}
        Notes
        -----
        The design of the filters is optimized for the value `L = 8`.

        The `pre_pad` flag is particularly useful when cropping bigger images
        because this does not introduce border effects inherent to padding.
        """

    _doc_scattering = \
    """Apply the scattering transform

       Parameters
       ----------
       input : {array}
           An input `{array}` of size `(B, M, N)`.

       Raises
       ------
       RuntimeError
           In the event that the input does not have at least two dimensions,
           or the tensor is not contiguous, or the tensor is not of the
           correct spatial size, padded or not.
       TypeError
           In the event that the input is not of type `{array}`.

       Returns
       -------
       S : {array}
           Scattering transform of the input. If `out_type` is set to
           `'array'` (or if it is not availabel for this frontend), this is
           a{n} `{array}` of shape `(B, C, M1, N1)` where `M1 = M // 2 ** J`
           and `N1 = N // 2 ** J`. The `C` is the number of scattering
           channels calculated. If `out_type` is `'list'`, the output is a
           list of dictionaries, with each dictionary corresponding to a
           scattering coefficient and its meta information. The actual
           coefficient is contained in the `'coef'` key, while other keys hold
           additional information, such as `'j'` (the scale of the filter
           used), and `'theta'` (the angle index of the filter used).
    """


    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''

        param_out_type = cls._doc_param_out_type if cls._doc_has_out_type else ''
        attr_out_type = cls._doc_attr_out_type if cls._doc_has_out_type else ''

        cls.__doc__ = ScatteringBase2D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call,
            instantiation=instantiation,
            param_shape=param_shape,
            attrs_shape=attrs_shape,
            param_out_type=param_out_type,
            attr_out_type=attr_out_type,
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        cls.scattering.__doc__ = ScatteringBase2D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


__all__ = ['ScatteringBase2D']
