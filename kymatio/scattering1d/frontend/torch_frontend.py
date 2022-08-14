import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
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
        self.load_filters()
        return super().scattering(x)


ScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D']
