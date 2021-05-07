import tensorflow as tf
import warnings

from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering import timefrequency_scattering
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D


class ScatteringTensorFlow1D(ScatteringTensorFlow, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, T=None, vectorize=True, out_type='array',
            pad_mode='reflect', max_pad_factor=2, backend='tensorflow',
            name='Scattering1D'):
        ScatteringTensorFlow.__init__(self, name=name)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average,
                oversampling, T, vectorize, out_type, pad_mode, max_pad_factor,
                backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

    def scattering(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        if not self.average and self.out_type == 'array' and self.vectorize:
            raise ValueError("Options average=False, out_type='array' and "
                             "vectorize=True are mutually incompatible. "
                             "Please set out_type to 'list' or vectorize to "
                             "False.")

        if not self.vectorize:
            warnings.warn("The vectorize option is deprecated and will be "
                          "removed in version 0.3. Please set "
                          "out_type='list' for equivalent functionality.",
                          DeprecationWarning)

        batch_shape = tf.shape(x)[:-1]
        signal_shape = tf.shape(x)[-1:]

        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling,
                         vectorize=self.vectorize,
                         size_scattering=size_scattering,
                         out_type=self.out_type)

        if self.out_type == 'array' and self.vectorize:
            scattering_shape = tf.shape(S)[-2:]
            new_shape = tf.concat((batch_shape, scattering_shape), 0)

            S = tf.reshape(S, new_shape)
        elif self.out_type == 'array' and not self.vectorize:
            for k, v in S.items():
                # NOTE: Have to get the shape for each one since we may have
                # average == False.
                scattering_shape = tf.shape(v)[-2:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)

                S[k] = tf.reshape(v, new_shape)
        elif self.out_type == 'list':
            for x in S:
                scattering_shape = tf.shape(x['coef'])[-1:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)

                x['coef'] = tf.reshape(x['coef'], new_shape)

        return S


ScatteringTensorFlow1D._document()


class TimeFrequencyScatteringTensorFlow1D(TimeFrequencyScatteringBase1D,
                                          ScatteringTensorFlow1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=1, T=None, F=None,
                 average=True, average_fr=None, oversampling=0,
                 oversampling_fr=None, aligned=True, resample_filters_fr=True,
                 out_type="array", pad_mode='zero', max_pad_factor=2,
                 backend="tensorflow"):
        if average_fr is None:
            average_fr = average
        if oversampling_fr is None:
            oversampling_fr = oversampling
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, oversampling_fr, aligned,
            resample_filters_fr, pad_mode)

        # Second-order scattering object for the time variable
        vectorize = True # for compatibility, will be removed in 0.3
        max_order_tm = 2
        _out_type = out_type if out_type != "array-like" else "array"
        ScatteringTensorFlow1D.__init__(
            self, J, shape, Q, max_order_tm, average, oversampling, T,
            vectorize, _out_type, pad_mode, max_pad_factor, backend)
        self.out_type = _out_type

        TimeFrequencyScatteringBase1D.build(self)

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not (self.average and self.average_fr) and 'array' in self.out_type:
            raise ValueError("Options average=False and out_type='array' "
                             "are mutually incompatible. "
                             "Please set out_type='list'.")

        if not self.out_type in ('array', 'list', 'array-like'):
            raise RuntimeError("The out_type must be one of: array, list, "
                               "array-like.")

        signal_shape = tf.shape(x)[-1:]
        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

        S = timefrequency_scattering(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.sc_freq,
            average=self.average,
            average_global=self.average_global,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            oversampling_fr=self.oversampling_fr,
            aligned=self.aligned,
            out_type=self.out_type,
            pad_mode=self.pad_mode)
        return S

TimeFrequencyScatteringTensorFlow1D._document()


__all__ = ['ScatteringTensorFlow1D', 'TimeFrequencyScatteringTensorFlow1D']
