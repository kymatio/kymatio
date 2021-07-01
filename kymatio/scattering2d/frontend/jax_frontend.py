from ...frontend.jax_frontend import ScatteringJax
from .numpy_frontend import ScatteringNumPy2D
from .base_frontend import ScatteringBase2D

class ScatteringJax1D(ScatteringJax, ScatteringNumPy2D, ScatteringBase2D):
    # This class inherits the attribute "frontend" from ScatteringJax
    # It overrides the __init__ function present in ScatteringNumPy2D 
    # in order to add the default argument for backend and call the
    # ScatteringJax.__init__
    # Through ScatteringBase2D._instantiate_backend the jax backend will
    # be loaded


    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, vectorize=True, out_type='array', backend='jax'):
        
        ScatteringJax.__init__(self)
        ScatteringBase2D.__init__(self, J, shape, Q, max_order, average,
                oversampling, vectorize, out_type, backend)
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

