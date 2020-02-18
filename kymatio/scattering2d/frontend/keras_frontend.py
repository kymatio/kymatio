from kymatio.tensorflow import Scattering2D as ScatteringTensorFlow2D

from tensorflow.keras.layers import Layer
from tensorflow.python.framework import tensor_shape


class ScatteringKeras2D(Layer):
    def __init__(self, J, L=8, max_order=2, pre_pad=False):
        super(ScatteringKeras2D, self).__init__()
        self.J = J
        self.L = L
        self.max_order = max_order
        self.pre_pad = pre_pad

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-2:])
        self.S = ScatteringTensorFlow2D(J=self.J, shape=shape,
                                        L=self.L, max_order=self.max_order,
                                        pre_pad=self.pre_pad)
        super(ScatteringKeras2D, self).build(input_shape)

    def call(self, x):
        return self.S(x)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        m0 = input_shape[-2] // 2 ** self.J
        m1 = input_shape[-1] // 2 ** self.J
        nc = (self.L ** 2) * self.J * (self.J - 1) // 2 + self.L * self.J + 1
        output_shape = [input_shape[0], nc, m0, m1]
        return tensor_shape.TensorShape(output_shape)
