from ...frontend.jax_frontend import ScatteringJax
from .numpy_frontend import ScatteringNumPy1D
from .base_frontend import ScatteringBase1D

class ScatteringJax1D(ScatteringJax, ScatteringNumPy1D, ScatteringBase1D):
    
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, vectorize=True, out_type='array', backend='jax'):
        ScatteringJax.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average,
                oversampling, vectorize, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

