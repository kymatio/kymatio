from ...frontend.keras_frontend import ScatteringKeras
from ...scattering1d.frontend.base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase

from kymatio.tensorflow import Scattering1D as ScatteringTensorFlow1D
from kymatio.tensorflow import TimeFrequencyScattering as TimeFrequencyScatteringTensorFlow

from tensorflow.python.framework import tensor_shape


class ScatteringKeras1D(ScatteringKeras, ScatteringBase1D):
    def __init__(self, J, Q=1, T=None, max_order=2, oversampling=0):
        ScatteringKeras.__init__(self)
        ScatteringBase1D.__init__(self, J, None, Q=Q, T=T, max_order=max_order,
            oversampling=oversampling, out_type='array')

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        self.S = ScatteringTensorFlow1D(J=self.J, shape=shape,
            Q=self._Q, T=self._T, max_order=self.max_order,
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
    def __init__(self, J, J_fr, Q, T=None, stride=None, Q_fr=1, F=None,
            stride_fr=None, format='time'):
        ScatteringKeras.__init__(self)
        TimeFrequencyScatteringBase.__init__(self, J, J_fr, Q, None, T, stride,
            Q_fr, F, stride_fr, 'array', format)

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        # The inner TensorFlow transform builds every (time and frequency)
        # filterbank itself, so the Keras layer only stores configuration and
        # delegates the heavy lifting to ``self.S`` -- mirroring
        # ScatteringKeras1D/2D. ``format`` must be forwarded, otherwise a
        # ``format='joint'`` layer would silently build a ``'time'`` transform.
        self.S = TimeFrequencyScatteringTensorFlow(J=self.J, J_fr=self.J_fr,
            Q=self._Q, shape=shape, T=self._T, stride=self._stride,
            Q_fr=self.Q_fr, F=self._F, stride_fr=self._stride_fr,
            format=self.format)
        ScatteringKeras.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        # The time axis is subsampled by 2 ** log2_stride (= log2(T)), not by
        # 2 ** J; global averaging collapses it to a single sample.
        if self.S.average == 'global':
            n_time = 1
        else:
            k = max(self.S.log2_stride, 0)
            n_time = self.S.ind_end[k] - self.S.ind_start[k]
        if self.format == 'joint':
            # (batch, n_jtfs, n_freq, time). ``meta()['n']`` already excludes
            # the zeroth-order path that joint + 'array' output does not emit,
            # so its length is the number of stacked joint coefficients. Every
            # joint path shares the same (averaged/subsampled) frequency axis.
            n_jtfs = len(self.S.meta()['n'])
            n_freq = self.S._N_padded_fr // 2 ** max(self.S.log2_stride_fr, 0)
            output_shape = [input_shape[0], n_jtfs, n_freq, n_time]
        else:
            # (batch, n_coefficients, time).
            output_shape = [input_shape[0], self.S.output_size(), n_time]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        # Only "static" parameters that determine the filterbank are
        # serialized; T, stride, F and stride_fr are dynamic (see gh-1053).
        # ``self._Q`` (not the ``Q`` property, which returns a tuple) is used so
        # that ``from_config(get_config())`` round-trips faithfully.
        keys = ["J", "J_fr", "Q_fr", "format"]
        config = {key: getattr(self, key) for key in keys}
        config["Q"] = self._Q
        return config


ScatteringKeras1D._document()
TimeFrequencyScatteringKeras._document()
