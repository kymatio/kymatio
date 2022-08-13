import warnings

from ...frontend.numpy_frontend import ScatteringNumPy
from .base_frontend import ScatteringBase1D


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2,
            oversampling=0, out_type='array', backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order,
                oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

ScatteringNumPy1D._document()


__all__ = ['ScatteringNumPy1D']
