import tensorflow as tf
import warnings

from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase


class ScatteringTensorFlow1D(ScatteringTensorFlow, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, oversampling=0,
            out_type='array', backend='tensorflow', name='Scattering1D'):
        ScatteringTensorFlow.__init__(self, name=name)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order,
                oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)


ScatteringTensorFlow1D._document()


class TimeFrequencyScatteringTensorFlow(
        ScatteringTensorFlow, TimeFrequencyScatteringBase):
    def __init__(self, *, J, J_fr, shape, Q, T=None, oversampling=0, Q_fr=1,
            F=None, oversampling_fr=0, out_type='array', backend='tensorflow', name='TimeFrequencyScattering'):
        ScatteringTensorFlow.__init__(self,name=name)
        TimeFrequencyScatteringBase.__init__(self, J=J, J_fr=J_fr, shape=shape,
            Q=Q, T=T, oversampling=oversampling, Q_fr=Q_fr, F=F,
            oversampling_fr=oversampling_fr, out_type=out_type, backend=backend)
        TimeFrequencyScatteringBase._instantiate_backend(
            self, 'kymatio.scattering1d.backend.')
        TimeFrequencyScatteringBase.build(self)
        TimeFrequencyScatteringBase.create_filters(self)
    



__all__ = ['ScatteringTensorFlow1D', 'TimeFrequencyScatteringTensorFlow']



