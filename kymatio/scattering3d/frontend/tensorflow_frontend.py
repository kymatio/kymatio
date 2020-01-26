# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

__all__ = ['HarmonicScattering3DTensorFlow']


import tensorflow as tf
from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ..core.scattering3d import scattering3d
from .base_frontend import ScatteringBase3D


class HarmonicScatteringTensorFlow3D(ScatteringTensorFlow, ScatteringBase3D):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2,
            rotation_covariant=True, method='integral', points=None,
            integral_powers=(0.5, 1., 2.), backend='tensorflow', name='HarmonicScattering3D'):
        ScatteringTensorFlow.__init__(self, name=name)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order,
                                  rotation_covariant, method, points,
                                  integral_powers, backend)
        self.build()


    def build(self):
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)

    def scattering(self, input_array):
        methods = ['integral']
        if (not self.method in methods):
            raise (ValueError('method must be in {}'.format(methods)))

        if self.method == 'integral':
            self.averaging =lambda x: self.backend.compute_integrals(x, self.integral_powers)

        with tf.name_scope('scattering') as scope:
            batch_shape = tuple(input_array.shape[:-3])
            signal_shape = tuple(input_array.shape[-3:])

            input_array = tf.reshape(input_array, (-1,) + signal_shape)

            S = scattering3d(input_array, filters=self.filters,
                             rotation_covariant=self.rotation_covariant,
                             L=self.L, J=self.J, max_order=self.max_order,
                             backend=self.backend, averaging=self.averaging)

            scattering_shape = tuple(S.shape[1:])

            S = tf.reshape(S, batch_shape + scattering_shape)

            return S

    @tf.Module.with_name_scope
    def __call__(self, x):
        """
        The forward pass of 3D solid harmonic scattering

        Parameters
        ----------
        input_array: tensorflow tensor
            input of size (batchsize, M, N, O)

        Returns
        -------
        output: tuple | tensorflow tensor
            if max_order is 1 it returns a torch tensor with the
            first order scattering coefficients
            if max_order is 2 it returns a torch tensor with the
            first and second order scattering coefficients,
            concatenated along the feature axis
        """
        try:
            x = tf.convert_to_tensor(x)
        except ValueError:
            raise TypeError('The input should be convertible to a '
                            'TensorFlow Tensor.')

        if len(x.shape) < 3:
            raise RuntimeError('Input tensor should have at least three '
                               'dimensions.')

        if (x.shape[-1] != self.O or x.shape[-2] != self.N or x.shape[-3] != self.M):
            raise RuntimeError(
                'Tensor must be of spatial size (%i, %i, %i).' % (
                    self.M, self.N, self.O))

        return self.scattering(x)
