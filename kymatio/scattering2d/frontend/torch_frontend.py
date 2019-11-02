__all__ = ['Scattering2DTorch']

import torch
import torch.nn as nn

from .base_frontend import Scattering2DBase
from kymatio.scattering2d.core.scattering2d import scattering2d
from ...frontend.torch_frontend import ScatteringTorch


class Scattering2DTorch(ScatteringTorch, Scattering2DBase):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend=None):
        ScatteringTorch.__init__(self)
        Scattering2DBase.__init__(self, J, shape, L, max_order, pre_pad, backend)
        self.backend_setup()
        Scattering2DBase.build(self)
        Scattering2DBase.create_filters(self)

        self.register_filters()


    def backend_setup(self):
        if not self.backend:
            from ..backend.torch_backend import backend
            self.backend = backend
        elif not self.backend.name.startswith('torch'):
            raise RuntimeError('This backend is not supported.')
        

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""

        # Create the filters
        n = 0
        for c, phi in self.phi.items():
            if isinstance(c, int):
                self.phi[c] = torch.from_numpy(self.phi[c]).unsqueeze(-1) # add a trailing singleton dimension to mark
                # it as non-complex
                self.register_buffer('tensor' + str(n), self.phi[c])
                n += 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if isinstance(k, int):
                    self.psi[j][k] = torch.from_numpy(v).unsqueeze(-1) # add a trailing singleton dimension to mark it
                    # as non-complex
                    self.register_buffer('tensor' + str(n), self.psi[j][k])
                    n += 1

    def scattering(self, input):
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        n = 0
        buffer_dict = dict(self.named_buffers())
        for c, phi in self.phi.items():
            if isinstance(c, int):
                self.phi[c] =  buffer_dict['tensor' + str(n)]
                n += 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if isinstance(k, int):
                    self.psi[j][k] = buffer_dict['tensor' + str(n)]
                    n += 1

        return scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi, self.max_order)

    def forward(self, input):
        """Forward pass of the scattering.

        Parameters
        ----------
        input : tensor
           Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
           arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
            :math:`n_2` is the number of
           input channels.

        Returns
        -------
        S : tensor
           Scattering of the input, a tensor with k+3 dimensions :math:`(n_1, ...,n_k, D, Md, Nd)`
           where :math:`D` corresponds to a new channel dimension and :math:`(Md, Nd)` are
           downsampled sizes by a factor :math:`2^J`. Currently, k=2 is hardcoded.

        """
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous!')

        if (input.size(-1) != self.N or input.size(-2) != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i)!' % (self.M, self.N))

        if (input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded))

        return self.scattering(input)

    def loginfo(self):
        return 'Torch frontend is used.'

