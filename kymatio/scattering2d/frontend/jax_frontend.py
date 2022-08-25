from ...frontend.jax_frontend import ScatteringJax
from .numpy_frontend import ScatteringNumPy2D
from .base_frontend import ScatteringBase2D

class ScatteringJax2D(ScatteringJax, ScatteringNumPy2D):
    # This class inherits the attribute "frontend" from ScatteringJax
    # It overrides the __init__ function present in ScatteringNumPy2D 
    # in order to add the default argument for backend and call the
    # ScatteringJax.__init__
    # Through ScatteringBase2D._instantiate_backend the jax backend will
    # be loaded


    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend='jax', out_type='array'):
        
        ScatteringJax.__init__(self)
        ScatteringBase2D.__init__(self, J, shape, L, max_order, pre_pad,
                backend, out_type)
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

ScatteringJax2D._document()
