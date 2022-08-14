import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from .base_frontend import ScatteringBase1D


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=None,
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
        for level in range(len(self.phi_f['levels'])):
            self.phi_f['levels'][level] = torch.from_numpy(
                self.phi_f['levels'][level]).float().view(-1, 1)
            self.register_buffer('tensor' + str(n), self.phi_f['levels'][level])
            n += 1
        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(
                    psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1
        for psi_f in self.psi2_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(
                    psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0

        for level in range(len(self.phi_f['levels'])):
            self.phi_f['levels'][level] = buffer_dict['tensor' + str(n)]
            n += 1

        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1

        for psi_f in self.psi2_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1

    def scattering(self, x):
        ScatteringBase1D._check_runtime_args(self)
        ScatteringBase1D._check_input(self, x)

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        self.load_filters()

        S = scattering1d(x, self.backend, self.psi1_f, self.psi2_f, self.phi_f,\
                         max_order=self.max_order, average=self.average, pad_left=self.pad_left, pad_right=self.pad_right,
                        ind_start=self.ind_start, ind_end=self.ind_end, oversampling=self.oversampling)

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


ScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D']
