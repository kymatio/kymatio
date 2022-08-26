import warnings

from ...frontend.numpy_frontend import ScatteringNumPy
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase


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


class TimeFrequencyScatteringNumPy(
        ScatteringNumPy1D, TimeFrequencyScatteringBase):
    def __init__(self, *, J, J_fr, shape, Q, T=None, oversampling=0, Q_fr=1,
            F=None, oversampling_fr=0, out_type='array', backend='numpy'):
        ScatteringNumPy.__init__(self)
        TimeFrequencyScatteringBase.__init__(self, J=J, J_fr=J_fr, shape=shape,
            Q=Q, T=T, oversampling=oversampling, Q_fr=Q_fr, F=F,
            oversampling_fr=oversampling_fr, out_type=out_type, backend=backend)
        ScatteringBase1D._instantiate_backend(
            self, 'kymatio.scattering1d.backend.')
        TimeFrequencyScatteringBase.build(self)
        TimeFrequencyScatteringBase.create_filters(self)

TimeFrequencyScatteringNumPy._document()

__all__ = ['ScatteringNumPy1D', 'TimeFrequencyScatteringNumPy']
