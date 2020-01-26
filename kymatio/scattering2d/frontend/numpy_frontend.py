from ...frontend.numpy_frontend import ScatteringNumPy
from kymatio.scattering2d.core.scattering2d import scattering2d
from .base_frontend import ScatteringBase2D
import numpy as np

class ScatteringNumPy2D(ScatteringNumPy, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase2D.__init__(self, J, shape, L, max_order, pre_pad, backend)
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

    def scattering(self, input):
        if not type(input) is np.ndarray:
            raise TypeError('The input should be a NumPy array.')

        if len(input.shape) < 2:
            raise RuntimeError('Input array must have at least two dimensions.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('NumPy array must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded array must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]

        input = input.reshape((-1,) + signal_shape)

        S = scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi, self.max_order,
                         vectorize=self.vectorize)

        scattering_shape = S.shape[-3:]

        S = S.reshape(batch_shape + scattering_shape)

        return S

__all__ = ['ScatteringNumPy2D']
