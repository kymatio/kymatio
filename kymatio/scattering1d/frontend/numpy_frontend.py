import math
from ...frontend.numpy_frontend import ScatteringNumPy
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering1d import timefrequency_scattering1d
from ..utils import precompute_size_scattering
from ...toolkit import pack_coeffs_jtfs
from .base_frontend import (ScatteringBase1D, TimeFrequencyScatteringBase1D,
                            _check_runtime_args_jtfs, _handle_args_jtfs)


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, out_type='array', pad_mode='reflect',
            max_pad_factor=2, analytic=False, normalize='l1-energy',
            r_psi=math.sqrt(.5), backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, pad_mode, max_pad_factor, analytic,
                normalize, r_psi, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

    def scattering(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if self.out_type == 'array' and not self.average:
            raise ValueError("out_type=='array' and average==False are mutually "
                             "incompatible. Please set out_type='list'.")

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        # get the arguments before calling the scattering
        # treat the arguments
        if self.average:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, self.T, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        S = scattering1d(x, self.pad_fn, self.backend.unpad, self.backend, self.J, self.log2_T, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average,
                         ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling,
                         size_scattering=size_scattering,
                         out_type=self.out_type)

        if self.out_type == 'array':
            scattering_shape = S.shape[-2:]
            new_shape = batch_shape + scattering_shape

            S = S.reshape(new_shape)
        else:
            for x in S:
                scattering_shape = x['coef'].shape[-1:]
                new_shape = batch_shape + scattering_shape

                x['coef'] = x['coef'].reshape(new_shape)

        return S

ScatteringNumPy1D._document()


class TimeFrequencyScatteringNumPy1D(TimeFrequencyScatteringBase1D,
                                     ScatteringNumPy1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=2, T=None, F=None,
                 implementation=None, average=True, average_fr=False,
                 oversampling=0, oversampling_fr=None, aligned=True,
                 sampling_filters_fr=('exclude', 'resample'), out_type="array",
                 out_3D=False, out_exclude=None, pad_mode='reflect',
                 pad_mode_fr='conj-reflect-zero', max_pad_factor=2,
                 max_pad_factor_fr=None, analytic=True, normalize='l1-energy',
                 r_psi=math.sqrt(.5), backend="numpy"):
        (oversampling_fr, normalize_tm, normalize_fr, r_psi_tm, r_psi_fr,
         max_order_tm, scattering_out_type) = (
            _handle_args_jtfs(oversampling, oversampling_fr, normalize, r_psi,
                              out_type))

        # Second-order scattering object for the time variable
        scattering_out_type = out_type.lstrip('dict:')
        ScatteringNumPy1D.__init__(
            self, J, shape, Q, T, max_order_tm, average, oversampling,
            scattering_out_type, pad_mode, max_pad_factor, analytic,
            normalize_tm, r_psi_tm, backend)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, implementation, average_fr, aligned,
            sampling_filters_fr, max_pad_factor_fr, pad_mode_fr, normalize_fr,
            r_psi_fr, oversampling_fr, out_3D, out_type, out_exclude)
        TimeFrequencyScatteringBase1D.build(self)

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        _check_runtime_args_jtfs(self.average, self.average_fr, self.out_type,
                                 self.out_3D)

        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)

        S = timefrequency_scattering1d(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.scf,
            self.pad_fn,
            average=self.average,
            average_global=self.average_global,
            average_global_phi=self.average_global_phi,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            oversampling_fr=self.oversampling_fr,
            aligned=self.aligned,
            out_type=self.out_type,
            out_3D=self.out_3D,
            out_exclude=self.out_exclude,
            pad_mode=self.pad_mode)
        if self.out_structure is not None:
            S = pack_coeffs_jtfs(S, self.meta(), self.out_structure,
                                 separate_lowpass=True,
                                 sampling_psi_fr=self.sampling_psi_fr)
        return S

    def scf_compute_padding_fr(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

    def scf_compute_J_pad_fr(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

TimeFrequencyScatteringNumPy1D._document()


__all__ = ['ScatteringNumPy1D', 'TimeFrequencyScatteringNumPy1D']
