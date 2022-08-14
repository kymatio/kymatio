import tensorflow as tf
import warnings

from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ..core.scattering1d import scattering1d
from .base_frontend import ScatteringBase1D


class ScatteringTensorFlow1D(ScatteringTensorFlow, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=None,
            oversampling=0, out_type='array', backend='tensorflow',
                 name='Scattering1D'):
        ScatteringTensorFlow.__init__(self, name=name)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

    def scattering(self, x):
        ScatteringBase1D._check_runtime_args(self)
        ScatteringBase1D._check_input(self, x)
        batch_shape = tf.shape(x)[:-1]
        signal_shape = tf.shape(x)[-1:]

        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

        S = scattering1d(x, self.backend, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling)

        if self.out_type == 'array':
            S = self.backend.concatenate([x['coef'] for x in S])
            scattering_shape = tf.shape(S)[-2:]
            new_shape = tf.concat((batch_shape, scattering_shape), 0)
            S = tf.reshape(S, new_shape)
        elif self.out_type == 'dict':
            S = {x['n']: x['coef'] for x in S}
            for k, v in S.items():
                # NOTE: Have to get the shape for each one since we may have
                # average == False.
                scattering_shape = tf.shape(v)[-1:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)
                S[k] = tf.reshape(v, new_shape)
        elif self.out_type == 'list':
            for x in S:
                scattering_shape = tf.shape(x['coef'])[-1:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)
                x['coef'] = tf.reshape(x['coef'], new_shape)

        return S


ScatteringTensorFlow1D._document()


__all__ = ['ScatteringTensorFlow1D']
