import torch

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering import timefrequency_scattering
from ..utils import precompute_size_scattering
from .base_frontend import (ScatteringBase1D, TimeFrequencyScatteringBase1D,
                            _check_runtime_args_jtfs)


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, out_type='array', pad_mode='reflect',
            max_pad_factor=2, register_filters=True, shorten=False,
            backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, pad_mode, max_pad_factor, shorten,
                backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        if register_filters:
            self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        # prepare for pytorch
        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = torch.from_numpy(
                    self.phi_f[k]).float()
                self.register_buffer('tensor' + str(n), self.phi_f[k])
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float()
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float()
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0

        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = buffer_dict['tensor' + str(n)]
                n += 1

        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

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

        self.load_filters()

        # get the arguments before calling the scattering
        # treat the arguments
        if self.average:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, self.T, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.log2_T, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling,
                         size_scattering=size_scattering,
                         out_type=self.out_type,
                         pad_mode=self.pad_mode)

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

ScatteringTorch1D._document()


class TimeFrequencyScatteringTorch1D(TimeFrequencyScatteringBase1D,
                                     ScatteringTorch1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=2, T=None, F=None,
                 average=True, average_fr=False, oversampling=0,
                 oversampling_fr=None, aligned=True, sampling_filters_fr='resample',
                 out_type="array", out_3D=False, out_exclude=None,
                 pad_mode='reflect', max_pad_factor=2, max_pad_factor_fr=None,
                 pad_mode_fr='conj-reflect-zero', shorten=False, backend="torch"):
        if oversampling_fr is None:
            oversampling_fr = oversampling
        # Second-order scattering object for the time variable
        max_order_tm = 2
        scattering_out_type = out_type.lstrip('dict:')
        ScatteringTorch1D.__init__(
            self, J, shape, Q, T, max_order_tm, average, oversampling,
            scattering_out_type, pad_mode, max_pad_factor,
            register_filters=False, shorten=shorten, backend=backend)

        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, oversampling_fr, aligned,
            sampling_filters_fr, max_pad_factor_fr, pad_mode_fr,
            out_3D, out_type, out_exclude)
        TimeFrequencyScatteringBase1D.build(self)
        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n_final = self._register_filters(self, ('phi_f', 'psi1_f', 'psi2_f'))
        self._register_filters(self.sc_freq,
                               ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_down'),
                               n0=n_final)

    def _register_filters(self, obj, filter_names, n0=0):
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if isinstance(p_f, dict):
                for k in p_f:
                    if isinstance(k, int):
                        if isinstance(p_f[k], list):
                            for k_sub in range(len(p_f[k])):
                                p_f[k][k_sub] = torch.from_numpy(
                                    p_f[k][k_sub]).float()
                                self.register_buffer(f'tensor{n}', p_f[k][k_sub])
                                n += 1
                        else:
                            p_f[k] = torch.from_numpy(p_f[k]).float()
                            self.register_buffer(f'tensor{n}', p_f[k])
                            n += 1
            else:  # list
                for p_f_sub in p_f:
                    for k in p_f_sub:
                        if isinstance(k, int):
                            p_f_sub[k] = torch.from_numpy(p_f_sub[k]).float()
                            self.register_buffer(f'tensor{n}', p_f_sub[k])
                            n += 1
        n_final = n
        return n_final

    def load_filters(self):
        """This function loads filters from the module's buffer """
        n_final = self._load_filters(self, ('phi_f', 'psi1_f', 'psi2_f'))
        self._load_filters(self.sc_freq,
                           ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_down'),
                           n0=n_final)

    def _load_filters(self, obj, filter_names, n0=0):
        buffer_dict = dict(self.named_buffers())
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if isinstance(p_f, dict):
                for k in p_f:
                    if isinstance(k, int):
                        if isinstance(p_f[k], list):
                            for k_sub in range(len(p_f[k])):
                                p_f[k][k_sub] = buffer_dict[f'tensor{n}']
                                n += 1
                        else:
                            p_f[k] = buffer_dict[f'tensor{n}']
                            n += 1
            else:  # list
                for p_f_sub in p_f:
                    for k in p_f_sub:
                        if isinstance(k, int):
                            p_f_sub[k] = buffer_dict[f'tensor{n}']
                            n += 1
        n_final = n
        return n_final

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        _check_runtime_args_jtfs(self.average, self.average_fr, self.out_type,
                                 self.out_3D)

        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)

        self.load_filters()

        S = timefrequency_scattering(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.sc_freq,
            average=self.average,
            average_global=self.average_global,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            oversampling_fr=self.oversampling_fr,
            aligned=self.aligned,
            out_type=self.out_type,
            out_3D=self.out_3D,
            out_exclude=self.out_exclude,
            pad_mode=self.pad_mode)
        return S

    def sc_freq_compute_padding_fr(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

    def sc_freq_compute_J_pad(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

TimeFrequencyScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D', 'TimeFrequencyScatteringTorch1D']
