from ...frontend.base_frontend import ScatteringBase
from ..filter_bank import (solid_harmonic_filter_bank, gaussian_filter_bank,
nd_gabor)


class ScatteringBaseHarmonic3D(ScatteringBase):
    """ Abstract module implementing the scattering transform in 2D. Actual
        implementations of the Scattering2D inherits from its methods and
        attributes. The scattering transform computes two wavelet transform
        followed by modulus non-linearity. It can be summarized as::

            S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]

        for::

            S_J^0 x = x * phi_J
            S_J^1 x = [|x * psi^1_lambda| * phi_J]_lambda
            S_J^2 x = [||x * psi^1_lambda| * psi^2_mu| * phi_J]_{lambda, mu}

        where * denotes the convolution (in space), phi_J is a lowpass
        filter, psi^1_lambda is a family of bandpass
        filters and psi^2_mu is another family of bandpass filters.
        Only Morlet filters are used in this implementation.
        Convolutions are efficiently performed in the Fourier domain.

        Example
        -------
            # 1) Define a Scattering2D object as:
                s = Scattering2D(J, shape=(M, N))
            #    where (M, N) is the image size and 2**J the scale of the scattering
            # 2) Forward on an input Tensor x of shape B x M x N,
            #     where B is the batch size.
                result_s = s(x)

        Parameters
        ----------
        J : int
            Log-2 of the scattering scale.
        shape : tuple of ints
            Spatial support (M, N) of the input.
        L : int, optional
            Number of angles used for the wavelet transform. Defaults to `8`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be either
            `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            Controls the padding: if set to False, a symmetric padding is applied
            on the signal. If set to True, the software will assume the signal was
            padded externally. Defaults to `False`.
        backend : object, optional
            Controls the backend which is combined with the frontend.

        Attributes
        ----------
        J : int
            Log-2 of the scattering scale.
        shape : tuple of int
            Spatial support (M, N) of the input.
        L : int, optional
            Number of angles used for the wavelet transform.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`.
        pre_pad : boolean
            Controls the padding: if set to False, a symmetric padding is applied
            on the signal. If set to True, the software will assume the signal was
            padded externally.
        Psi : dictionary
            Contains the wavelets filters at all resolutions. See
            filter_bank.filter_bank for an exact description.
        Phi : dictionary
            Contains the low-pass filters at all resolutions. See
            filter_bank.filter_bank for an exact description.
        M_padded, N_padded : int
             Spatial support of the padded input.

        Notes
        -----
        The design of the filters is optimized for the value L = 8.

        pre_pad is particularly useful when cropping bigger images because
        this does not introduce border effects inherent to padding.
        """
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2,
                 rotation_covariant=True, method='standard', points=None,
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
        self.M, self.N, self.P = self.shape

    def create_filters(self):
        self.filters = solid_harmonic_filter_bank(
            self.M, self.N, self.P, self.J, self.L, self.sigma_0)

        self.gaussian_filters = gaussian_filter_bank(
            self.M, self.N, self.P, self.J + 1, self.sigma_0)

class ScatteringBase3D(ScatteringBase):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2,
                 rotation_covariant=True, method='standard', points=None,
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
        self.M, self.N, self.P = self.shape

    def create_filters(self):


        self.filters = gabor_nd(self.shape, orientation, self.J, xi0=3* np.pi
                / 4, sigma0=self.sigma_0,slant=.5, remove_dc=True, ifftshift=True):

__all__ = ['ScatteringBase3D']
