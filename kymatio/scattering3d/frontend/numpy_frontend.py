# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

__all__ = ['HarmonicScattering3DNumpy']

from ...frontend.numpy_frontend import ScatteringNumpy

from kymatio.scattering3d.filter_bank import solid_harmonic_filter_bank, gaussian_filter_bank
from kymatio.scattering3d.core.scattering3d import scattering3d




class HarmonicScattering3DNumpy(ScatteringNumpy):
    """3D Solid Harmonic scattering .

    This class implements solid harmonic scattering on an input 3D image.
    For details see https://arxiv.org/abs/1805.00571.

    Instantiates and initializes a 3d solid harmonic scattering object.

    Parameters
    ----------
    J: int
        number of scales
    shape: tuple of int
        shape (M, N, O) of the input signal
    L: int, optional
        Number of l values. Defaults to 3.
    sigma_0: float, optional
        Bandwidth of mother wavelet. Defaults to 1.
    max_order: int, optional
        The maximum order of scattering coefficients to compute. Must be
        either 1 or 2. Defaults to 2.
    rotation_covariant: bool, optional
        if set to True the first order moduli take the form:

        $\\sqrt{\\sum_m (x \\star \\psi_{j,l,m})^2)}$

        if set to False the first order moduli take the form:

        $x \\star \\psi_{j,l,m}$

        The second order moduli change analogously
        Defaut: True
    method: string, optional
        specifies the method for obtaining scattering coefficients
        ("standard","local","integral"). Default: "standard"
    points: array-like, optional
        List of locations in which to sample wavelet moduli. Used when
        method == 'local'
    integral_powers: array-like
        List of exponents to the power of which moduli are raised before
        integration. Used with method == 'standard', method == 'integral'

    """
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='standard', points=None,
                 integral_powers=(0.5, 1., 2.), backend=None):
        super(HarmonicScattering3DNumpy, self).__init__()
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

        self.build()

    def build(self):
        self.M, self.N, self.O = self.shape

        if not self.backend:
            from ..backend.numpy_backend import backend  # is imported like a module and not a class?
            self.backend = backend
        elif self.backend.name[0:5] != 'numpy':
            raise RuntimeError('This backend is not supported.')

        self.filters = solid_harmonic_filter_bank(
                            self.M, self.N, self.O, self.J, self.L, self.sigma_0)
        self.gaussian_filters = gaussian_filter_bank(
                                self.M, self.N, self.O, self.J + 1, self.sigma_0)

        methods = ['standard', 'local', 'integral']
        if (not self.method in methods):
            raise (ValueError('method must be in {}'.format(methods)))
        if self.method == 'integral':\
            self.averaging =lambda x,j: self.backend.compute_integrals(self.backend.fft(x, inverse=True)[...,0],
                                                                       self.integral_powers)
        elif self.method == 'local':
            self.averaging = lambda x,j:\
                self.backend._compute_local_scattering_coefs(x,
                        self.gaussian_filters[j+1], self.points)
        elif self.method == 'standard':
            self.averaging = lambda x, j:\
                self.backend._compute_standard_scattering_coefs(x,
                        self.gaussian_filters[j], self.J, self.backend.subsample)


    def scattering(self, input_array):
        return scattering3d(input_array, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L,
                            J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)

    def loginfo(self):
        return 'Numpy front end is used.'
