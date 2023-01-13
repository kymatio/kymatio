from ...frontend.keras_frontend import ScatteringKeras
from ...scattering1d.frontend.base_frontend import ScatteringBase1D

from kymatio.tensorflow import Scattering1D as ScatteringTensorFlow1D

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

ScatteringKeras1D._document()
