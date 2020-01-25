import torch

from .base_frontend import ScatteringBase2D
from kymatio.scattering2d.core.scattering2d import scattering2d
from ...frontend.torch_frontend import ScatteringTorch
import torch.nn as nn


class filters_(nn.Module):
    def __init__(self, J, f):
        super(filters_, self).__init__()
        self.j = J
        self.filters = nn.ParameterList([nn.Parameter(torch.from_numpy(f[j]).unsqueeze(-1)) for j in range(len(f))])

    def __getitem__(self, key):
        if key == 'j':
            return self.j
        if key == 'filters':
            return self.filters

class ScatteringTorch2D(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)
        self.register_filters()


    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters
        self.phi_ = filters_(self.phi['j'], self.phi['filters'])
        self.psi_ = nn.ModuleList()
        #f = self.phi['filters']
        #self.phi['filters'] = nn.ParameterList([nn.Parameter(torch.from_numpy(f[j]).unsqueeze(-1)) for j in range(len(f))])

        for i in range(len(self.psi)):
            psi = filters_(self.psi[i]['j'], self.psi[i]['filters'])
            self.psi_.append(psi)
            #nn.ModuleDict(psi[i].j)
            #f = self.psi[i]['filters']
            #self.psi[i]['filters'] = nn.ParameterList([nn.Parameter(torch.from_numpy(f[j]).unsqueeze(-1)) for j in range(len(f))])


    def scattering(self, input):
        """Forward pass of the scattering.

            Parameters
            ----------
            input : tensor
                Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
                arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
                :math:`n_2` is the number of input channels.

            Raises
            ------
            RuntimeError
                In the event that the input does not have at least two
                dimensions, or the tensor is not contiguous, or the tensor is
                not of the correct spatial size, padded or not.
            TypeError
                In the event that the input is not a Torch tensor.

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
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]

        input = input.reshape((-1,) + signal_shape)

        S = scattering2d(input, self.pad, self.unpad, self.backend, self.J,
                            self.L, self.phi_, self.psi_, self.max_order)

        scattering_shape = S.shape[-3:]

        S = S.reshape(batch_shape + scattering_shape)

        return S


__all__ = ['ScatteringTorch2D']
