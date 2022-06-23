from ...frontend.numpy_frontend import ScatteringNumPy
from kymatio.scattering3d.core.scattering3d import scattering3d
from .base_frontend import ScatteringBase3D
import numpy as np


class HarmonicScatteringNumPy3D(ScatteringNumPy, ScatteringBase3D):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='integral', points=None,
                 integral_powers=(0.5, 1., 2.), backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order,
                                  rotation_covariant, method, points,
                                  integral_powers, backend)

        self.build()

    def build(self):
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)

    def scattering(self, input_array):
        self.backend.input_checks(input_array)

        if input_array.ndim < 3:
            raise RuntimeError('Input tensor must have at least three '
                               'dimensions.')

        if (input_array.shape[-1] != self.O or input_array.shape[-2] != self.N
            or input_array.shape[-3] != self.M):
            raise RuntimeError(
                'Tensor must be of spatial size (%i, %i, %i).' % (
                    self.M, self.N, self.O))

        batch_shape = input_array.shape[:-3]
        signal_shape = input_array.shape[-3:]

        input_array = input_array.reshape((-1,) + signal_shape)
        self.averaging = lambda x: self.backend.compute_integrals(x, self.integral_powers)

        S = scattering3d(input_array, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L,
                            J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)


        scattering_shape = S.shape[1:]

        S = S.reshape(batch_shape + scattering_shape)

        return S


HarmonicScatteringNumPy3D._document()


__all__ = ['HarmonicScatteringNumPy3D']
