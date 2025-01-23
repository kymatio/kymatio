from ...frontend.keras_frontend import ScatteringKeras
from ...scattering1d.frontend.base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase

from kymatio.tensorflow import Scattering1D as ScatteringTensorFlow1D

from kymatio.tensorflow import TimeFrequencyScattering as TimeFrequencyScatteringTensorFlow
from tensorflow.python.framework import tensor_shape


class ScatteringKeras1D(ScatteringKeras, ScatteringBase1D):
    def __init__(self, J, Q=1, T=None, max_order=2, oversampling=0):
        ScatteringKeras.__init__(self)
        self.J = J
        self._Q = Q
        self._T = T
        self._max_order = 2
        self._oversampling = 0
        self.out_type = 'array'
        self.backend = None

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        self.S = ScatteringTensorFlow1D(J=self.J, shape=shape,
            Q=self._Q, T=self._T, max_order=self._max_order,
            oversampling=self._oversampling)
        ScatteringKeras.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        nc = self.S.output_size()
        k0 = max(self.J - self._oversampling, 0)
        ln = self.S.ind_end[k0] - self.S.ind_start[k0]
        output_shape = [input_shape[0], nc, ln]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        keys = ["J", "Q", "max_order", "oversampling"]
        return {key: getattr(self, key) for key in keys}


class TimeFrequencyScatteringKeras(ScatteringKeras, TimeFrequencyScatteringBase):
    def __init__(self, 
        J,
        J_fr,
        Q,
        shape,
        T=None,
        stride=None,
        Q_fr=1,
        F=None,
        stride_fr=None,
        out_type="array",
        backend="tensorflow",):

        ScatteringKeras.__init__(self)

        self.J=J,
        self.J_fr=J_fr,
        self.Q=Q,
        self.shape=shape,
        self.T=T,
        self.stride=stride,
        self.Q_fr=Q_fr,
        self.F=F,
        self.stride_fr=stride_fr,
        self.out_type=out_type,
        self.backend=backend,

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        self.S = TimeFrequencyScatteringTensorFlow(
        J=self.J,
        J_fr=self.J_fr,
        Q=self.Q,
        shape=self.shape,
        T=self.T,
        stride=self.stride,
        Q_fr=self.Q_fr,
        F=self.F,
        stride_fr=self.stride_fr)
        ScatteringKeras.build(self, input_shape)

    #TODO: how do we implement this without #839 implemented?
    #right now uses ScatteringKeras1D calculation
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        nc = self.S.output_size()
        k0 = max(self.J - self._oversampling, 0)
        ln = self.S.ind_end[k0] - self.S.ind_start[k0]
        output_shape = [input_shape[0], nc, ln]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        keys = ["J", "J_fr", "Q", "max_order", "Q_fr"]
        return {key: getattr(self, key) for key in keys}


ScatteringKeras1D._document()
TimeFrequencyScatteringKeras._document()
