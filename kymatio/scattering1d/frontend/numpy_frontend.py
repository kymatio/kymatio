# Authors: Mathieu Andreux, Joakim Anden, Edouard Oyallon
# Scientific Ancestry: Joakim Anden, Mathieu Andreux, Vincent Lostanlen

import warnings

from ...frontend.numpy_frontend import ScatteringNumPy
from ..core.scattering1d import scattering1d
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, vectorize=True, out_type='array', backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average,
                oversampling, vectorize, out_type, backend)
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

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

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
            scattering_shape = S.shape[-2:]
            new_shape = batch_shape + scattering_shape

            S = S.reshape(new_shape)
        elif self.out_type == 'array' and not self.vectorize:
            for k, v in S.items():
                # NOTE: Have to get the shape for each one since we may have
                # average == False.
                scattering_shape = v.shape[-2:]
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
