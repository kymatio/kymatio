import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering import timefrequency_scattering
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, T=None, vectorize=True, out_type='array',
            pad_mode='reflect', max_pad_factor=2, register_filters=True,
            backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average,
                oversampling, T, vectorize, out_type, pad_mode, max_pad_factor,
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
                # view(-1, 1).repeat(1, 2) because real numbers!
                self.phi_f[k] = torch.from_numpy(
                    self.phi_f[k]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), self.phi_f[k])
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    # view(-1, 1).repeat(1, 2) because real numbers!
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float().view(-1, 1)
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    # view(-1, 1).repeat(1, 2) because real numbers!
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float().view(-1, 1)
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

        self.load_filters()

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

ScatteringTorch1D._document()


class TimeFrequencyScatteringTorch1D(TimeFrequencyScatteringBase1D,
                                     ScatteringTorch1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=1, T=None, F=None,
                 average=True, average_fr=None, oversampling=0,
                 oversampling_fr=None, aligned=True, resample_filters_fr=True,
                 out_type="array", pad_mode='zero', max_pad_factor=2,
                 backend="torch"):
        if average_fr is None:
            average_fr = average
        if oversampling_fr is None:
            oversampling_fr = oversampling
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, oversampling_fr, aligned,
            resample_filters_fr, pad_mode)

        # Second-order scattering object for the time variable
        vectorize = True # for compatibility, will be removed in 0.3
        max_order_tm = 2
        _out_type = out_type if out_type != "array-like" else "array"
        ScatteringTorch1D.__init__(
            self, J, shape, Q, max_order_tm, average, oversampling, T,
            vectorize, _out_type, pad_mode, max_pad_factor,
            register_filters=False, backend=backend)
        self.out_type = _out_type

        TimeFrequencyScatteringBase1D.build(self)
        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n_final = self._register_filters(self, ('phi_f', 'psi1_f'))
        self._register_filters(self.sc_freq,
                               ('phi_f', 'psi1_f_up', 'psi1_f_down'), n0=n_final)

    def _register_filters(self, obj, filter_names, n0=0):
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if isinstance(p_f, dict):
                for k in p_f:
                    if isinstance(k, int):
                        p_f[k] = torch.from_numpy(p_f[k]).float().view(-1, 1)
                        self.register_buffer(f'tensor{n}', p_f[k])
            else:  # list
                for p_f_sub in p_f:
                    for k in p_f_sub:
                        if isinstance(k, int):
                            p_f_sub[k] = torch.from_numpy(p_f_sub[k]
                                                          ).float().view(-1, 1)
                            self.register_buffer(f'tensor{n}', p_f_sub[k])
            n += 1
        n_final = n
        return n_final

    def load_filters(self):
        """This function loads filters from the module's buffer """
        n_final = self._load_filters(self, ('phi_f', 'psi1_f'))
        self._load_filters(self.sc_freq,
                           ('phi_f', 'psi1_f_up', 'psi1_f_down'), n0=n_final)

    def _load_filters(self, obj, filter_names, n0=0):
        buffer_dict = dict(self.named_buffers())
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if isinstance(p_f, dict):
                for k in p_f:
                    if isinstance(k, int):
                        p_f[k] = buffer_dict[f'tensor{n}']
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

        if not (self.average and self.average_fr) and 'array' in self.out_type:
            raise ValueError("Options average=False and out_type='array' "
                             "are mutually incompatible. "
                             "Please set out_type='list'.")

        if not self.out_type in ('array', 'list', 'array-like'):
            raise RuntimeError("The out_type must be one of: array, list, "
                               "array-like.")

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
            pad_mode=self.pad_mode)
        return S

TimeFrequencyScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D', 'TimeFrequencyScatteringTorch1D']
