import warnings

from ...frontend.numpy_frontend import ScatteringNumPy
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering import timefrequency_scattering
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, T=None, vectorize=True, out_type='array',
            pad_mode='reflect', max_pad_factor=2, backend='numpy'):
        ScatteringNumPy.__init__(self)
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

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, self.T, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.log2_T, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling,
                         vectorize=self.vectorize,
                         size_scattering=size_scattering,
                         out_type=self.out_type,
                         pad_mode=self.pad_mode)

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


class TimeFrequencyScatteringNumPy(TimeFrequencyScatteringBase, ScatteringNumPy1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=1, T=None, F=None,
                 average=True, average_fr=None, oversampling=0,
                 oversampling_fr=None, aligned=True, resample_psi_fr=True,
                 resample_phi_fr=True, out_type="array", pad_mode='zero',
                 max_pad_factor=2, backend="numpy"):
        if average_fr is None:
            average_fr = average
        if oversampling_fr is None:
            oversampling_fr = oversampling
        TimeFrequencyScatteringBase.__init__(
            self, J_fr, Q_fr, F, average_fr, oversampling_fr, aligned,
            resample_psi_fr, resample_phi_fr, pad_mode)

        # Second-order scattering object for the time variable
        vectorize = True # for compatibility, will be removed in 0.3
        max_order_tm = 2
        _out_type = out_type if out_type != "array-like" else "array"
        ScatteringNumPy1D.__init__(
            self, J, shape, Q, max_order_tm, average, oversampling, T,
            vectorize, _out_type, pad_mode, max_pad_factor, backend)
        self.out_type = _out_type

        TimeFrequencyScatteringBase.build(self)

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not self.average and self.out_type == 'array':
            raise ValueError("Options average=False and out_type='array' "
                             "are mutually incompatible. "
                             "Please set out_type='list'.")

        if not self.out_type in ('array', 'list', 'array-like'):
            raise RuntimeError("The out_type must be one of: array, list, "
                               "array-like.")

        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)

        S = timefrequency_scattering(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.sc_freq,
            average=self.average,
            average_fr=self.average_fr,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            oversampling_fr=self.oversampling_fr,
            aligned=self.aligned,
            out_type=self.out_type,
            pad_mode=self.pad_mode)
        return S

TimeFrequencyScatteringNumPy._document()


__all__ = ['ScatteringNumPy1D', 'TimeFrequencyScatteringNumPy']
