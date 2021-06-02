from ...frontend.keras_frontend import ScatteringKeras
from ...scattering2d.frontend.base_frontend import ScatteringBase2D

from ...tensorflow import Scattering2D as ScatteringTensorFlow2D

from tensorflow.python.framework import tensor_shape


class ScatteringKeras2D(ScatteringKeras, ScatteringBase2D):
    def __init__(self, J, L=8, max_order=2, pre_pad=False):
        ScatteringKeras.__init__(self)
        ScatteringBase2D.__init__(self, J, None, L, max_order, pre_pad,
                                  'array')

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-2:])
        self.S = ScatteringTensorFlow2D(J=self.J, shape=shape,
                                        L=self.L, max_order=self.max_order,
                                        pre_pad=self.pre_pad)
        ScatteringKeras.build(self, input_shape)
        # NOTE: Do not call ScatteringBase2D.build here since that will
        # recreate the filters already in self.S

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        m0 = input_shape[-2] // 2 ** self.J
        m1 = input_shape[-1] // 2 ** self.J
        nc = (self.L ** 2) * self.J * (self.J - 1) // 2 + self.L * self.J + 1
        output_shape = [input_shape[0], nc, m0, m1]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        keys = ["J", "L", "max_order", "pre_pad"]
        return {key: getattr(self, key) for key in keys}

ScatteringKeras2D._document()
