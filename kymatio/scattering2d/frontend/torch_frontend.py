import torch

from .base_frontend import ScatteringBase2D
from ...scattering2d.core.scattering2d import scattering2d
from ...frontend.torch_frontend import ScatteringTorch


class ScatteringTorch2D(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend='torch', out_type='array'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

        if pre_pad:
            # Need to cast to complex in Torch
            self.pad = lambda x: x.reshape(x.shape + (1,))

        self.register_filters()

    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        n = 0

        for phi_level in self.phi['levels']:
            self.register_single_filter(phi_level, n)
            n = n + 1

        for psi in self.psi:
            for psi_level in psi['levels']:
                self.register_single_filter(psi_level, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        buffer_dict = dict(self.named_buffers())

        n = 0

        phis = {k: v for k, v in self.phi.items() if k!='levels'}
        phis['levels'] = []
        for phi_level in self.phi['levels']:
            phis['levels'].append(self.load_single_filter(n, buffer_dict))
            n = n + 1

        psis = [{} for _ in range(len(self.psi))]
        for j in range(len(self.psi)):
            psis[j] = {k: v for k, v in self.psi[j].items() if k!='levels'}
            psis[j]['levels'] = []
            for psi_level in self.psi[j]['levels']:
                psis[j]['levels'].append(self.load_single_filter(n, buffer_dict))
                n = n + 1

        return phis, psis

    def scattering(self, input):
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.shape[-1] or input.shape[-2] != self.shape[-2]) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.shape[0], self.shape[1]))

        if (input.shape[-1] != self._N_padded or input.shape[-2] != self._M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self._M_padded, self._N_padded))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        phi, psi = self.load_filters()

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]

        input = input.reshape((-1,) + signal_shape)

        S = scattering2d(input, self.pad, self.unpad, self.backend, self.J,
                            self.L, phi, psi, self.max_order, self.out_type)

        if self.out_type == 'array':
            scattering_shape = S.shape[-3:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            scattering_shape = S[0]['coef'].shape[-2:]
            new_shape = batch_shape + scattering_shape

            for x in S:
                x['coef'] = x['coef'].reshape(new_shape)

        return S


ScatteringTorch2D._document()


__all__ = ['ScatteringTorch2D']
