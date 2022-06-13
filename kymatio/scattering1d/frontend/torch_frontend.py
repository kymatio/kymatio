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
        return super().scattering(x)


ScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D']
