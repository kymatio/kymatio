import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from .base_frontend import ScatteringBase1D


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, out_type='array', backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
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
                    self.phi_f[k]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), self.phi_f[k])
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float().view(-1, 1)
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
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
        self.load_filters()
        ScatteringBase1D._check_runtime_args(self)
        ScatteringBase1D._check_input(self, x)
        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)

        S = scattering1d(x, self.backend, self.psi1_f, self.psi2_f, self.phi_f,\
                         max_order=self.max_order, average=self.average, pad_left=self.pad_left, pad_right=self.pad_right,
                        ind_start=self.ind_start, ind_end=self.ind_end, oversampling=self.oversampling)

        n_kept_dims = 1 + (self.out_type == "dict")
        for n, path in enumerate(S):
            S[n]['coef'] = self.backend.reshape_output(
                path['coef'], batch_shape, n_kept_dims=n_kept_dims)

        if self.out_type == 'array':
            return self.backend.concatenate([path['coef'] for path in S], dim=-2)
        elif self.out_type == 'dict':
            return {path['n']: path['coef'] for path in S}
        elif self.out_type == 'list':
            return list(map(lambda path: path.pop('n')), S)


ScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D']
