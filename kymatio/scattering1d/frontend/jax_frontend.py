from ...frontend.jax_frontend import ScatteringJax
from .numpy_frontend import ScatteringNumPy1D
from .base_frontend import ScatteringBase1D

class ScatteringJax1D(ScatteringJax, ScatteringNumPy1D):
    # This class inherits the attribute "frontend" from ScatteringJax
    # It overrides the __init__ function present in ScatteringNumPy1D
    # in order to add the default argument for backend and call the
    # ScatteringJax.__init__
    # Through ScatteringBase1D._instantiate_backend the jax backend will
    # be loaded


    def __init__(self, J, shape, Q=1, T=None, stride=None, max_order=2,
            oversampling=0, out_type='array', backend='jax'):

        ScatteringJax.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, stride, max_order,
                oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
