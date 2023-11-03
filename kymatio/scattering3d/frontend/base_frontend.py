from ...frontend.base_frontend import ScatteringBase
from ..filter_bank import solid_harmonic_filter_bank, gaussian_filter_bank


class ScatteringBase3D(ScatteringBase):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2,
                 rotation_covariant=True, method='integral', points=None,
                 integral_powers=(0.5, 1., 2.), backend=None):
        super(ScatteringBase3D, self).__init__()
        self.J = J
        self.shape = shape
        self.L = L
        self.sigma_0 = sigma_0

        self.max_order = max_order
        self.rotation_covariant = rotation_covariant
        self.method = method
        self.points = points
        self.integral_powers = integral_powers
        self.backend = backend

    def build(self):
        self.M, self.N, self.O = self.shape

    def create_filters(self):
        self.filters = solid_harmonic_filter_bank(
            self.M, self.N, self.O, self.J, self.L, self.sigma_0)

        self.gaussian_filters = gaussian_filter_bank(
            self.M, self.N, self.O, self.J + 1, self.sigma_0)

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    _doc_shape = 'M, N, O'

    _doc_class = \
    r"""The 3D solid harmonic scattering transform

        This class implements solid harmonic scattering on a 3D input image.
        For details see https://arxiv.org/abs/1805.00571.{frontend_paragraph}

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N, O = 32, 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a HarmonicScattering3D object.
            S = HarmonicScattering3D(J, (M, N, O))

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}

        Parameters
        ----------
        J: int
            Number of scales.
        shape: tuple of ints
            Shape `(M, N, O)` of the input signal
        L: int, optional
            Number of `l` values. Defaults to `3`.
        sigma_0: float, optional
            Bandwidth of mother wavelet. Defaults to `1`.
        max_order: int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        rotation_covariant: bool, optional
            If set to `True` the first-order moduli take the form:

            $\sqrt{{\sum_m (x \star \psi_{{j,l,m}})^2)}}$

            if set to `False` the first-order moduli take the form:

            $x \star \psi_{{j,l,m}}$

            The second order moduli change analogously. Defaults to `True`.
        method: string, optional
            Specifies the method for obtaining scattering coefficients.
            Currently, only `'integral'` is available. Defaults to `'integral'`.
        integral_powers: array-like
            List of exponents to the power of which moduli are raised before
            integration.
        """

    _doc_scattering = \
    """Apply the scattering transform

       Parameters
       ----------
       input_array: {array}
           Input of size `(batch_size, M, N, O)`.

       Returns
       -------
       output: {array}
           If max_order is `1` it returns a{n} `{array}` with the first-order
           scattering coefficients. If max_order is `2` it returns a{n}
           `{array}` with the first- and second- order scattering
           coefficients, concatenated along the feature axis.
    """

    @classmethod
    def _document(cls):
        cls.__doc__ = ScatteringBase3D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call.format(x="x"),
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        # Sphinx will not show docstrings for inherited methods, so we add a
        # dummy method here that will just call the super.
        if not "scattering" in cls.__dict__:
            def _scattering(self, x):
                return super(cls, self).scattering(x)

            setattr(cls, "scattering", _scattering)

        cls.scattering.__doc__ = ScatteringBase3D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


__all__ = ['ScatteringBase3D']
