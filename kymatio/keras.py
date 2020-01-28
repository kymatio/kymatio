from kymatio.tensorflow import Scattering1D as ScatteringTensorFlow1D
from kymatio.tensorflow import Scattering2D as ScatteringTensorFlow2D

from tensorflow.keras.layers import Layer
from tensorflow.python.framework import tensor_shape


class Scattering1D(Layer):
    def __init__(self, J, Q=1, max_order=2, oversampling=0):
        super(Scattering1D, self).__init__()
        self.J = J
        self.Q = Q
        self.max_order = max_order
        self.oversampling = oversampling

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        self.S = ScatteringTensorFlow1D(J=self.J, shape=shape,
                                        Q=self.Q, max_order=self.max_order,
                                        oversampling=self.oversampling)
        super(Scattering1D, self).build(input_shape)

    def call(self, x):
        return self.S(x)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        nc = self.S.output_size()
        k0 = max(self.J - self.oversampling, 0)
        ln = self.S.ind_end[k0] - self.S.ind_start[k0]
        output_shape = [input_shape[0], nc, ln]
        return tensor_shape.TensorShape(output_shape)


class Scattering2D(Layer):
    def __init__(self, J, L=8, max_order=2, pre_pad=False):
        super(Scattering2D, self).__init__()
        self.J = J
        self.L = L
        self.max_order = max_order
        self.pre_pad = pre_pad

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-2:])
        self.S = ScatteringTensorFlow2D(J=self.J, shape=shape,
                                        L=self.L, max_order=self.max_order,
                                        pre_pad=self.pre_pad)
        super(Scattering2D, self).build(input_shape)

    def call(self, x):
        return self.S(x)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        m0 = input_shape[-2] // 2 ** self.J
        m1 = input_shape[-1] // 2 ** self.J
        nc = (self.L ** 2) * self.J * (self.J - 1) // 2 + self.L * self.J + 1
        output_shape = [input_shape[0], nc, m0, m1]
        return tensor_shape.TensorShape(output_shape)
