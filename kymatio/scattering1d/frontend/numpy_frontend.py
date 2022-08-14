import warnings

from ...frontend.numpy_frontend import ScatteringNumPy
from ..core.scattering1d import scattering1d
from .base_frontend import ScatteringBase1D


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=None,
            oversampling=0, out_type='array', backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

    def scattering(self, x):
        ScatteringBase1D._check_runtime_args(self)
        ScatteringBase1D._check_input(self, x)

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        S = scattering1d(x, self.backend, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling)

        if self.out_type == 'array':
            S = self.backend.concatenate([x['coef'] for x in S])
            scattering_shape = S.shape[-2:]
            new_shape = batch_shape + scattering_shape
            S = S.reshape(new_shape)
        elif self.out_type == 'dict':
            S = {x['n']: x['coef'] for x in S}
            for k, v in S.items():
                # NOTE: Have to get the shape for each one since we may have
                # average == False.
                scattering_shape = v.shape[-1:]
                new_shape = batch_shape + scattering_shape
                S[k] = v.reshape(new_shape)
        elif self.out_type == 'list':
            for x in S:
                scattering_shape = x['coef'].shape[-1:]
                new_shape = batch_shape + scattering_shape
                x['coef'] = x['coef'].reshape(new_shape)

        return S


ScatteringNumPy1D._document()


__all__ = ['ScatteringNumPy1D']
