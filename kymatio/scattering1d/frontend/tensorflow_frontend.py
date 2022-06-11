import tensorflow as tf
import warnings

from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from ..core.scattering1d import scattering1d
from .base_frontend import ScatteringBase1D


class ScatteringTensorFlow1D(ScatteringTensorFlow, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
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
        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape, n_inserted_dims=1)

        S = scattering1d(x, self.backend, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling, out_type=self.out_type)

        if self.out_type == 'array':
            S = self.backend.reshape_output(S, batch_shape, n_kept_dims=2)
        elif self.out_type == 'dict':
            for k, v in S.items():
                S[k] = self.backend.reshape_output(v, batch_shape, n_kept_dims=2)
        elif self.out_type == 'list':
            for path in S:
                path['coef'] = self.backend.reshape_output(
                    path['coef'], batch_shape, n_kept_dims=1)

        return S


ScatteringTensorFlow1D._document()


__all__ = ['ScatteringTensorFlow1D']
