import tensorflow as tf
from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ...scattering2d.core.scattering2d import scattering2d
from .base_frontend import ScatteringBase2D

class ScatteringTensorFlow2D(ScatteringTensorFlow, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend='tensorflow', name='Scattering2D', out_type='array'):
        ScatteringTensorFlow.__init__(self, name)
        ScatteringBase2D.__init__(self, J, shape, L, max_order, pre_pad,
                backend, out_type)
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

        if pre_pad:
            # Need to cast to complex in TF
            self.pad = lambda x: tf.cast(x, tf.complex64)


    def scattering(self, input):
        with tf.name_scope('scattering') as scope:
            try:
                input = tf.convert_to_tensor(input)
            except ValueError:
                raise TypeError('The input should be convertible to a '
                                'TensorFlow Tensor.')

            if len(input.shape) < 2:
                raise RuntimeError('Input tensor should have at least two '
                                   'dimensions.')

            if (input.shape[-1] != self.shape[-1] or input.shape[-2] != self.shape[-2]) and not self.pre_pad:
                raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.shape[0], self.shape[1]))

            if (input.shape[-1] != self._N_padded or input.shape[-2] != self._M_padded) and self.pre_pad:
                raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self._M_padded, self._N_padded))

            if not self.out_type in ('array', 'list'):
                raise RuntimeError("The out_type must be one of 'array' or 'list'.")

            # Use tf.shape to get the dynamic shape of the tf.Tensors at
            # execution time.
            batch_shape = tf.shape(input)[:-2]
            signal_shape = tf.shape(input)[-2:]

            # NOTE: Cannot simply concatenate these using + since they are
            # tf.Tensors and that would add their values.
            input = tf.reshape(input, tf.concat(((-1,), signal_shape), 0))

            S = scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi,
                             self.max_order, self.out_type)

            if self.out_type == 'array':
                scattering_shape = tf.shape(S)[-3:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)

                S = tf.reshape(S, new_shape)
            else:
                scattering_shape = tf.shape(S[0]['coef'])[-2:]
                new_shape = tf.concat((batch_shape, scattering_shape), 0)

                for x in S:
                    x['coef'] = tf.reshape(x['coef'], new_shape)

            return S


ScatteringTensorFlow2D._document()


__all__ = ['ScatteringTensorFlow2D']
