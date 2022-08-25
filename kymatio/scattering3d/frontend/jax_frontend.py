from ...frontend.jax_frontend import ScatteringJax
from .numpy_frontend import HarmonicScatteringNumPy3D
from .base_frontend import ScatteringBase3D


class HarmonicScatteringJax3D(ScatteringJax, HarmonicScatteringNumPy3D):
    # This class inherits the attribute "frontend" from ScatteringJax
    # It overrides the __init__ function present in ScatteringNumPy3D 
    # in order to add the default argument for backend and call the
    # ScatteringJax.__init__
    # Through ScatteringBase3D._instantiate_backend the jax backend will
    # be loaded
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='integral', points=None,
                 integral_powers=(0.5, 1., 2.), backend='jax'):
        ScatteringJax.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order,
                                  rotation_covariant, method, points,
                                  integral_powers, backend)

        self.build()

    def build(self):
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)

HarmonicScatteringJax3D._document()
